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
    datetime,
)  # Used in _build_summary_rows, _process_transactions...

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
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
        is_cash_symbol,
        get_conversion_rate,  # Used by _build_summary_rows
        calculate_irr,  # Used by _build_summary_rows
        CashFlowLedger,  # Used by _build_summary_rows
        safe_sum,  # Used by _calculate_aggregate_metrics
        get_currency_symbol_from_code,
        get_historical_rate_via_usd_bridge,  # Added for dividend history
        get_cash_flows_for_mwr,  # Used for overall MWR
        calculate_indicated_dividend,  # NEW: Unified dividend logic
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
    logging.debug("Starting Numba-optimized transaction processing (Array-based)...")

    if transactions_df.empty:
        # Shape must match the full return below: holdings, realized, dividends,
        # commissions, taxes, ignored indices, ignored reasons, transfer costs, warned.
        return {}, {}, {}, {}, {}, set(), {}, {}, False

    if account_cash_mode_map is None:
        account_cash_mode_map = {}

    # --- Normalize Columns EARLY ---
    # Critical: Normalize strings BEFORE deduplication to ensure matching works
    transactions_df["Symbol"] = transactions_df["Symbol"].astype(str).str.strip()
    # Normalize cash-like symbols (e.g. 'Cash ($)') to the designated CASH_SYMBOL_CSV ($CASH)
    transactions_df["Symbol"] = transactions_df["Symbol"].apply(
        lambda x: CASH_SYMBOL_CSV if is_cash_symbol(x) else x
    )

    transactions_df["Account"] = transactions_df["Account"].astype(str).str.strip()
    transactions_df["Type"] = (
        transactions_df["Type"].astype(str).str.strip().str.lower()
    )

    # Create ID mappings
    all_symbols_list = list(transactions_df["Symbol"].unique())
    # Ensure $CASH is always in the symbol map for auto-cash accounting
    if CASH_SYMBOL_CSV not in all_symbols_list:
        all_symbols_list.append(CASH_SYMBOL_CSV)
    all_symbols = all_symbols_list
    all_accounts = set(transactions_df["Account"].unique())
    if "To Account" in transactions_df.columns:
        all_accounts.update(
            transactions_df["To Account"].dropna().astype(str).str.strip().unique()
        )
    # Sorted, not set order: account ids decide the order of the holdings, so
    # of the summary's rows and its per-account metrics, and a set of strings
    # iterates differently in every process (hash seeding). Name order is also
    # the order the account picker shows (`_available_accounts`).
    all_accounts = sorted(all_accounts)
    all_currencies = (
        transactions_df["Local Currency"].astype(str).str.strip().str.upper().unique()
    )

    sym_map = {s: i for i, s in enumerate(all_symbols)}
    acc_map = {a: i for i, a in enumerate(all_accounts)}
    curr_map = {c: i for i, c in enumerate(all_currencies)}

    rev_sym_map = {i: s for s, i in sym_map.items()}
    rev_acc_map = {i: a for a, i in acc_map.items()}
    rev_curr_map = {i: c for c, i in curr_map.items()}

    n = len(transactions_df)
    num_syms = len(all_symbols)
    num_accs = len(all_accounts)
    # --- Prepare Data for Numba ---
    df = transactions_df.copy()

    # --- Deduplicate Split Transactions (Robustness Fix) ---
    # Splits currently apply to ALL accounts in the Numba loop.
    # Must ensure we don't process multiple splits for the same symbol/date (e.g. one from 'All Accounts' and one from 'E*TRADE').
    is_split_mask = df["Type"].isin(["split", "stock split"])

    if is_split_mask.any():
        # logging.debug("Deduplicating split transactions...")
        splits_df = df[is_split_mask].copy()
        other_df = df[~is_split_mask]

        # Ensure we can sort. Convert Date to datetime if not already (it should be)
        if not pd.api.types.is_datetime64_any_dtype(splits_df["Date"]):
            splits_df["Date"] = pd.to_datetime(splits_df["Date"], errors="coerce")

        # Priority:
        # 1. 'All Accounts' (assumed manual override/correct) -> Rank 0
        # 2. Others -> Rank 1
        splits_df["__split_priority"] = np.where(
            splits_df["Account"].str.lower() == "all accounts", 0, 1
        )

        # --- Fuzzy Deduplication (Month-Year) ---
        # Create a Year-Month integer for grouping (Safe for all environments)
        # Year * 100 + Month (e.g. 201406)
        splits_df["__split_ym"] = (
            splits_df["Date"].dt.year * 100 + splits_df["Date"].dt.month
        ).astype(np.int64)

        # Sort by Date, Symbol, Priority
        # We want to keep the highest priority (lowest rank) for each symbol/month
        splits_df.sort_values(by=["Date", "Symbol", "__split_priority"], inplace=True)

        # Drop duplicates based on Symbol and Year-Month
        deduped_splits = splits_df.drop_duplicates(
            subset=["__split_ym", "Symbol", "Split Ratio"], keep="first"
        ).drop(columns=["__split_priority", "__split_ym"])

        # Recombine and sort by date/index to maintain chronology
        # Force a copy and consolidation to avoid internal pandas BlockManager AssertionErrors
        df = pd.concat(
            [other_df.copy(), deduped_splits.copy()], ignore_index=True
        ).sort_values(by=["Date", "original_index"])
        # Ensure 'Date' and 'original_index' are used for sorting
        df.sort_values(by=["Date", "original_index"], inplace=True)
        # Force consolidation and new index to avoid Numba memory view issues
        df = df.reset_index(drop=True).copy()
        n = len(df)  # UPDATE: n must reflect the new size after deduplication
        # logging.debug(f"Transactions reduced from {len(transactions_df)} to {len(df)} after split deduplication.")

    sym_ids = df["Symbol"].map(sym_map).fillna(-1).astype(np.int64).values
    acc_ids = df["Account"].map(acc_map).fillna(-1).astype(np.int64).values

    type_map_dict = {
        "buy": TYPE_BUY,
        "sell": TYPE_SELL,
        "deposit": TYPE_DEPOSIT,
        "withdrawal": TYPE_WITHDRAWAL,
        "dividend": TYPE_DIVIDEND,
        "fee": TYPE_FEES,
        "fees": TYPE_FEES,
        "split": TYPE_SPLIT,
        "stock split": TYPE_SPLIT,
        "transfer": TYPE_TRANSFER,
        "short sell": TYPE_SHORT_SELL,
        "buy to cover": TYPE_BUY_TO_COVER,
        "tax": TYPE_TAX,
        "withholding tax": TYPE_TAX,
        "interest": TYPE_INTEREST,
        "spin off": TYPE_SPIN_OFF,
        "spin-off": TYPE_SPIN_OFF,
        "spinoff": TYPE_SPIN_OFF,
    }
    type_ids = (
        df["Type"]
        .str.lower()
        .str.strip()
        .map(type_map_dict)
        .fillna(TYPE_UNKNOWN)
        .astype(np.int64)
        .values
    )

    qtys = (
        pd.to_numeric(df["Quantity"], errors="coerce")
        .fillna(0.0)
        .astype(np.float64)
        .values
    )

    raw_prices = (
        pd.to_numeric(df["Price/Share"], errors="coerce")
        .fillna(0.0)
        .astype(np.float64)
        .values
    )
    raw_totals = (
        pd.to_numeric(df["Total Amount"], errors="coerce")
        .fillna(0.0)
        .astype(np.float64)
        .values
    )

    # --- Robust Quantity/Price Handling for Cash ---
    # For $CASH, if Total Amount is provided, we should trust it.
    # If Quantity is also provided and matches Total Amount, it means the user
    # entered the amount in both columns (common in manual entries).
    # In this case, we should set Price/Share to 1.0 to avoid squared values in calculations.
    cash_sym_mask = (df["Symbol"] == CASH_SYMBOL_CSV).values
    zero_qty_mask = np.abs(qtys) < 1e-9
    has_total_mask = np.abs(raw_totals) > 1e-9

    fix_cash_mask = cash_sym_mask & has_total_mask
    if np.any(fix_cash_mask):
        # Case 1: Qty is 0, use Total
        mask_qty_0 = fix_cash_mask & zero_qty_mask
        qtys[mask_qty_0] = np.abs(raw_totals[mask_qty_0])
        raw_prices[mask_qty_0] = 1.0

        # Case 2: Qty matches Total, force Price to 1.0
        # (This prevents Dividend logic from doing Qty * Price = Amount * Amount)
        mask_qty_match = (
            fix_cash_mask
            & (~zero_qty_mask)
            & (np.abs(qtys - np.abs(raw_totals)) < 1e-4)
        )
        raw_prices[mask_qty_match] = 1.0

    # Use Total Amount for types where Price/Share might be zero (Dividends, Tax, Fees, Interest)
    # OR for any Cash transaction with zero Price/Share OR zero Quantity
    income_expense_types = (
        (type_ids == TYPE_DIVIDEND)
        | (type_ids == TYPE_TAX)
        | (type_ids == TYPE_FEES)
        | (type_ids == TYPE_INTEREST)
    )
    mask_use_total = (
        (income_expense_types | cash_sym_mask)
        & ((np.abs(raw_prices) < 1e-9) | (np.abs(qtys) < 1e-9))
        & has_total_mask
    )
    raw_prices[mask_use_total] = raw_totals[mask_use_total]

    # A dividend states its payment in Total Amount whenever that column is
    # filled: it is the cash the broker actually paid. Shares x per-share rate
    # only reconstructs it — and on a reinvestment the share count is rounded,
    # so the product drifts from the payment (0.13 * 113.4363 = 14.75 for a
    # 14.63 dividend). Fall back to that product, then to a bare Price/Share,
    # for rows that carry no total. The historical engine's _dividend_amount
    # reads a row the same way, so the graph and the holdings table agree.
    mask_div_total = (type_ids == TYPE_DIVIDEND) & has_total_mask
    raw_prices[mask_div_total] = np.abs(raw_totals[mask_div_total])

    mask_div_calc = (type_ids == TYPE_DIVIDEND) & (~mask_use_total) & (~mask_div_total)
    raw_prices[mask_div_calc] = raw_prices[mask_div_calc] * np.abs(qtys[mask_div_calc])

    prices = raw_prices

    comms = (
        pd.to_numeric(df["Commission"], errors="coerce")
        .fillna(0.0)
        .astype(np.float64)
        .values
    )
    split_ratios = (
        pd.to_numeric(df["Split Ratio"], errors="coerce")
        .fillna(0.0)
        .astype(np.float64)
        .values
    )

    # --- Fallback: Use Quantity as Split Ratio if Split Ratio is 0 ---
    # Some importers (e.g. Sharebuilder/E*TRADE auto-imports) put the ratio in the Quantity column
    # while leaving the Split Ratio column empty/zero.
    mask_split_no_ratio = (
        (type_ids == TYPE_SPLIT)
        & (np.abs(split_ratios) <= 1e-9)
        & (np.abs(qtys) > 1e-9)
    )
    if np.any(mask_split_no_ratio):
        # logging.debug(f"Processing {np.sum(mask_split_no_ratio)} splits with missing ratio...")

        # Case A: Quantity is "Reasonable Ratio" (<= 20). Use as Ratio.
        mask_ratio_fallback = mask_split_no_ratio & (np.abs(qtys) <= 20.0)
        split_ratios[mask_ratio_fallback] = qtys[mask_ratio_fallback]

        # Case B: Quantity is "Target Total Quantity" (> 20). Encode as NEGATIVE value.
        # This signals the Numba loop to use "Target Quantity" logic instead of "Ratio" logic.
        mask_target_qty = mask_split_no_ratio & (np.abs(qtys) > 20.0)
        split_ratios[mask_target_qty] = -np.abs(qtys[mask_target_qty])

        # if np.any(mask_target_qty):
        #     logging.debug(f"Identified {np.sum(mask_target_qty)} splits where Quantity (>20) is treated as Target Total Quantity.")

    to_acc_ids = np.full(n, -1, dtype=np.int64)
    if "To Account" in df.columns:
        df["To Account"] = df["To Account"].astype(str).str.strip()
        to_acc_ids = df["To Account"].map(acc_map).fillna(-1).astype(np.int64).values

    df["Local Currency"] = df["Local Currency"].astype(str).str.strip().str.upper()
    invalid_curr = ~df["Local Currency"].isin(curr_map)
    if invalid_curr.any():
        if default_currency not in curr_map:
            curr_map[default_currency] = len(curr_map)
            rev_curr_map[len(curr_map) - 1] = default_currency
        df.loc[invalid_curr, "Local Currency"] = default_currency

    local_curr_ids = df["Local Currency"].map(curr_map).astype(np.int64).values

    dates = df["Date"].dt.date.values
    currs = df["Local Currency"].values
    fx_rates_hist = np.array(
        [historical_fx_lookup.get((d, c), 1.0) for d, c in zip(dates, currs)],
        dtype=np.float64,
    )

    shortable_sym_ids_set = set()
    for s in shortable_symbols:
        if s in sym_map:
            shortable_sym_ids_set.add(sym_map[s])

    nb_shortable = np.array(sorted(shortable_sym_ids_set), dtype=np.int64)

    cash_sym_id = sym_map.get(CASH_SYMBOL_CSV, -1)

    # --- Build acc_cash_modes array (0=Manual, 1=Auto) ---
    acc_cash_modes = np.zeros(num_accs, dtype=np.int64)
    # Normalize map keys to match stripped/normalized account names
    normalized_mode_map = {str(k).strip(): v for k, v in account_cash_mode_map.items()}
    for acc_name, mode_str in normalized_mode_map.items():
        if acc_name in acc_map and mode_str == "Auto":
            acc_cash_modes[acc_map[acc_name]] = 1

    final_state, transfer_costs_nb = _process_numba_core(
        sym_ids,
        acc_ids,
        type_ids,
        qtys,
        prices,
        comms,
        raw_totals,
        split_ratios,
        to_acc_ids,
        local_curr_ids,
        fx_rates_hist,
        nb_shortable,
        STOCK_QUANTITY_CLOSE_TOLERANCE,
        cash_sym_id,
        n,
        num_syms,
        num_accs,
        acc_cash_modes,
    )

    # --- Tags Aggregation (Outside Numba) ---
    tags_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    if "Tags" in df.columns:
        # Vectorized approach to collect tags would be complex, simple iteration is fine for metadata
        # or use groupby
        # Let's iterate over rows where Tags is not empty
        df_tags = df[df["Tags"].notna() & (df["Tags"] != "")]
        for _, row in df_tags.iterrows():
            sym = row["Symbol"]
            acc = row["Account"]
            t_str = str(row["Tags"])
            # Split by comma if multiple tags
            for t in t_str.split(","):
                cleaned_t = t.strip()
                if cleaned_t:
                    tags_map[(sym, acc)].add(cleaned_t)

            # Also handle transfer-in side?
            # If it's a transfer, the tag should technically follow?
            # For now, let's just stick to the account on the transaction record.

    holdings = {}
    overall_realized_gains_local = defaultdict(float)
    overall_dividends_local = defaultdict(float)
    overall_commissions_local = defaultdict(float)
    overall_taxes_local = defaultdict(float)

    # Unpack 3D Array
    # Iterate only over initialized entries
    # We can use np.where to find initialized entries (index 11 != -1)
    # But iterating num_syms * num_accs is fast if they are small.
    # If they are large, np.where is better.
    # Let's use simple loops for now, or np.argwhere.

    # Using np.argwhere on the currency channel
    active_indices = np.argwhere(final_state[:, :, 11] != -1.0)

    for idx in active_indices:
        s_id, a_id = idx
        val = final_state[s_id, a_id]

        sym = rev_sym_map[s_id]
        acc = rev_acc_map[a_id]
        curr = rev_curr_map[int(val[11])]

        holdings[(sym, acc)] = {
            "qty": val[0],
            "total_cost_local": val[1],
            "realized_gain_local": val[2],
            "dividends_local": val[3],
            "commissions_local": val[4],
            "local_currency": curr,
            "short_proceeds_local": val[5],
            "short_original_qty": val[6],
            "total_cost_invested_local": val[7],
            "cumulative_investment_local": val[8],
            "total_buy_cost_local": val[9],
            "total_cost_display_historical_fx": val[10],
            "realized_gain_display": val[12],
            "taxes_local": val[13],
            "tags": sorted(list(tags_map.get((sym, acc), set()))),
        }

        overall_realized_gains_local[curr] += val[2]
        overall_dividends_local[curr] += val[3]
        overall_commissions_local[curr] += val[4]
        overall_taxes_local[curr] += val[13]

    transfer_costs = {}
    orig_indices = df["original_index"].values
    # transfer_costs_nb is array of size n
    # Find indices where it is not nan
    valid_tc_indices = np.where(~np.isnan(transfer_costs_nb))[0]
    for idx in valid_tc_indices:
        if idx < len(orig_indices):
            transfer_costs[orig_indices[idx]] = transfer_costs_nb[idx]

    return (
        holdings,
        dict(overall_realized_gains_local),
        dict(overall_dividends_local),
        dict(overall_commissions_local),
        dict(overall_taxes_local),
        set(),
        {},
        transfer_costs,
        False,
    )


# --- REVISED: _build_summary_rows (uses MarketDataProvider) ---
class _HoldingRows:
    """
    The transaction rows `_build_summary_rows` filters for every holding,
    selected once per summary.

    Each holding used to mask the whole ledger by symbol, account and date: the
    same-day flows asked for the rows on the report date, and the last-trade
    price fallback for priced rows up to it, converting every row's date
    (`.dt.date`) each time. Filtering by date and price first and by symbol and
    account second selects the same rows, in the same order, with the same
    labels. A failure here is kept and raised again inside each holding's own
    handler, so those handlers log what they logged before.
    """

    def __init__(self, transactions_df: pd.DataFrame, report_date: date):
        self.df = transactions_df
        self.report_date = report_date
        self._memo: Dict[str, Tuple[bool, Any]] = {}

    def _once(self, key: str, build):
        if key not in self._memo:
            try:
                self._memo[key] = (True, build())
            except Exception as exc:
                self._memo[key] = (False, exc)
        ok, value = self._memo[key]
        if not ok:
            raise value
        return value

    def _days(self) -> pd.Series:
        return self._once("days", lambda: self.df["Date"].dt.date)

    def on_report_date(self) -> pd.DataFrame:
        """`df[df["Date"].dt.date == report_date]`"""
        return self._once(
            "on_report_date", lambda: self.df[self._days() == self.report_date]
        )

    def priced_by_report_date(self) -> pd.DataFrame:
        """Rows with a positive Price/Share dated on or before the report date."""

        def build():
            df = self.df
            return df[
                (df["Price/Share"].notna())
                & (pd.to_numeric(df["Price/Share"], errors="coerce") > 1e-9)
                & (self._days() <= self.report_date)
            ]

        return self._once("priced_by_report_date", build)


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
    portfolio_summary_rows: List[Dict[str, Any]] = []
    account_local_currency_map: Dict[str, str] = {}
    has_errors = False
    has_warnings = False

    # Normalize once so the filter is case/whitespace-insensitive — the
    # holdings dict uses the raw account string from transactions, which may
    # differ in case from the user's filter list.
    include_accounts_norm: Optional[Set[str]] = None
    if include_accounts:
        include_accounts_norm = {str(a).strip().upper() for a in include_accounts}

    if account_interest_rates is None:
        account_interest_rates = {}
    if interest_free_thresholds is None:
        interest_free_thresholds = {}

    logging.info(f"Calculating final portfolio summary rows in {display_currency}...")

    # --- BUG-12 FIX: Removed dead scaffold code that duplicated cash/stock separation ---

    # --- Separate cash from stock holdings ---
    cash_holdings_by_currency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stock_holdings: Dict[Tuple[str, str], Dict] = {}

    for holding_key, data in holdings.items():
        symbol, account = holding_key
        if is_cash_symbol(symbol):
            # Group cash holdings by their local currency for aggregation
            currency = data.get("local_currency", default_currency)
            cash_holdings_by_currency[currency].append({"account": account, **data})
        else:
            stock_holdings[holding_key] = data

    # Every holding's IRR flows come from one read of the ledger.
    cash_flow_ledger = CashFlowLedger(transactions_df)
    holding_rows = _HoldingRows(transactions_df, report_date)

    for holding_key, data in stock_holdings.items():
        symbol, account = holding_key
        current_qty = data.get("qty", 0.0)
        realized_gain_local = data.get("realized_gain_local", 0.0)
        dividends_local = data.get("dividends_local", 0.0)
        commissions_local = data.get("commissions_local", 0.0)
        taxes_local = data.get("taxes_local", 0.0)
        local_currency = data.get("local_currency", default_currency)
        current_total_cost_local = data.get("total_cost_local", 0.0)
        short_proceeds_local = data.get("short_proceeds_local", 0.0)
        total_cost_invested_local = data.get("total_cost_invested_local", 0.0)
        cumulative_investment_local = data.get("cumulative_investment_local", 0.0)
        total_buy_cost_local = data.get("total_buy_cost_local", 0.0)
        # NEW: Get historical cost in display currency
        total_cost_display_historical_fx_val = data.get(
            "total_cost_display_historical_fx", np.nan
        )
        realized_gain_display_from_holdings = data.get("realized_gain_display", np.nan)
        tags_list = data.get("tags", [])

        account_local_currency_map[account] = local_currency
        stock_data = current_stock_data.get(symbol, {})

        # --- Price Determination ---
        price_source = "Unknown"
        current_price_local = np.nan
        day_change_local = np.nan
        day_change_pct = np.nan

        # --- MODIFIED: Price Determination Logic ---
        if is_cash_symbol(symbol):
            price_source = "Internal (Cash)"
            current_price_local = 1.0
            day_change_local = 0.0
            day_change_pct = 0.0
        else:
            is_excluded = symbol in user_excluded_symbols
            current_price_local_raw = stock_data.get("price")
            is_yahoo_price_valid = (
                pd.notna(current_price_local_raw)
                and isinstance(current_price_local_raw, (int, float))
                and current_price_local_raw > 1e-9
            )

            if not is_excluded and is_yahoo_price_valid:
                price_source = "Yahoo API/Cache"
                current_price_local = float(current_price_local_raw)
                day_change_local_raw = stock_data.get("change")
                day_change_pct_raw = stock_data.get(
                    "changesPercentage"
                )  # This is % value from provider
                day_change_local = (
                    float(day_change_local_raw)
                    if pd.notna(day_change_local_raw)
                    else np.nan
                )
                day_change_pct = (
                    float(day_change_pct_raw)
                    if pd.notna(day_change_pct_raw)
                    else np.nan
                )
                logging.debug(
                    f"Price OK ({symbol}): Using Yahoo price {current_price_local}"
                )
            elif not is_excluded:
                # if symbol != "UNH": # Silence specific UNH warnings if needed or just remove log
                #      # logging.warning(
                #      #    f"Warning: Yahoo price invalid/missing for {symbol}. Trying fallbacks."
                #      # )
                #      pass
                price_source = "Yahoo Invalid"
                has_warnings = True
            elif is_excluded:
                logging.debug(
                    f"Info: Symbol {symbol} is excluded. Skipping Yahoo fetch, trying fallbacks."
                )
                price_source = "Excluded"
                has_warnings = True
        # --- END MODIFIED Price Determination ---

        # A published NAV sits between the market feed and the hand-entered
        # override. These holdings are on the exclusion list — they have no
        # ticker, so the block above never prices them — and the override below
        # is a number somebody typed once. The SEC's daily series is the same
        # kind of number kept current, and it is already what the graph uses,
        # so taking it here is what makes the two agree.
        if pd.isna(current_price_local) and published_navs:
            nav_row = published_navs.get(symbol)
            if nav_row is not None:
                nav_date, nav_value = nav_row
                current_price_local = float(nav_value)
                price_source = f"Published NAV ({nav_date})"
                # A fund publishes one NAV a day and no intraday change; leaving
                # these NaN keeps the day-change column empty rather than
                # inventing a zero move.
                day_change_local = np.nan
                day_change_pct = np.nan

        if pd.isna(current_price_local):
            manual_price = manual_prices_dict.get(symbol)
            if (
                manual_price is not None
                and pd.notna(manual_price)
                and isinstance(manual_price, (int, float))
                and manual_price > 0
            ):
                current_price_local = float(manual_price)
                price_source += " - Manual Fallback"
                logging.debug(
                    f"Info: Used MANUAL fallback price {current_price_local} for {symbol}/{account}"
                )
                day_change_local = np.nan
                day_change_pct = np.nan
            else:
                if symbol in manual_prices_dict:
                    logging.warning(
                        f"Warn: Manual price found for {symbol} but was invalid/zero ({manual_price}). Trying Last TX."
                    )
                else:
                    logging.debug(
                        f"Info: Manual price not found for {symbol}. Trying Last TX."
                    )

        if pd.isna(current_price_local):
            # If quantity is zero, we don't strictly need a price for valuation.
            # We can skip the noisy fallback/warning and just default to 0.
            if abs(current_qty) < 1e-9:
                current_price_local = 0.0
                price_source += " - Qty 0 (Skip)"
            else:
                price_source += (
                    " - No Manual" if "Manual Fallback" not in price_source else ""
                )
                try:
                    # Match either side of a transfer: shares may have arrived
                    # via a Transfer row where the priced leg is on the source
                    # account (Account=src, "To Account"=account).
                    priced = holding_rows.priced_by_report_date()
                    account_mask = priced["Account"] == account
                    if "To Account" in transactions_df.columns:
                        account_mask = account_mask | (priced["To Account"] == account)
                    fallback_tx = priced[
                        (priced["Symbol"] == symbol) & account_mask
                    ].copy()
                    if not fallback_tx.empty:
                        fallback_tx.sort_values(
                            by=["Date", "original_index"], inplace=True, ascending=True
                        )
                        last_tx_row = fallback_tx.iloc[-1]
                        last_tx_price = pd.to_numeric(
                            last_tx_row["Price/Share"], errors="coerce"
                        )
                        last_tx_date = last_tx_row["Date"].date()
                        if pd.notna(last_tx_price) and last_tx_price > 1e-9:
                            current_price_local = float(last_tx_price)
                            price_source += (
                                f" - Last TX ({last_tx_price:.2f}@{last_tx_date})"
                            )
                            logging.debug(
                                f"Info: Used Last TX fallback price {current_price_local} for {symbol}/{account}"
                            )
                        else:
                            logging.warning(
                                f"Warn: Fallback TX found for {symbol}/{account} but price invalid ({last_tx_price}). Using 0."
                            )
                            price_source += " - Last TX Invalid/Zero"
                            current_price_local = 0.0
                    else:
                        logging.warning(
                            f"Warn: No valid prior TX found for fallback for {symbol}/{account}. Using 0."
                        )
                        price_source += " - No Last TX"
                        current_price_local = 0.0
                except Exception as e_fallback:
                    logging.error(
                        f"ERROR during last TX fallback for {symbol}/{account}: {e_fallback}"
                    )
                    price_source += " - Fallback Error"
                    current_price_local = 0.0
            day_change_local = np.nan
            day_change_pct = np.nan

        if pd.isna(current_price_local):
            logging.error(
                f"ERROR: All price sources failed for {symbol}/{account}. Forcing 0."
            )
            current_price_local = 0.0
            if "Using 0" not in price_source and "Error" not in price_source:
                price_source += " - Forced Zero"
            has_warnings = True
        current_price_local = float(current_price_local)

        # --- Currency Conversion ---
        # Ensure fx_rate is defined before the try block for IRR
        # It's used for converting local currency values to display currency.
        # If this fails, many subsequent calculations will result in NaN.

        fx_rate = get_conversion_rate(
            local_currency, display_currency, current_fx_rates_vs_usd
        )
        if pd.isna(fx_rate):  # Keep critical error log if FX rate is NaN
            logging.error(
                f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for {symbol}/{account}."
            )
            has_errors = True
            fx_rate = np.nan

        # --- NEW: Unified Dividend Calculations (Forward-Looking) ---
        # We now use Indicated Annual Dividend (IAD) instead of Trailing 12 Months (TTM).
        # This aligns the dashboard summary metrics with the projected income charts.
        indicated_annual_dividend_rate_local = calculate_indicated_dividend(stock_data)

        dividend_yield_on_current_direct = stock_data.get("dividendYield")

        div_yield_on_cost_pct_local = np.nan
        div_yield_on_current_pct_local = np.nan
        est_annual_income_local = np.nan

        if indicated_annual_dividend_rate_local > 0:
            # Estimated Annual Income (Local) - Only for long positions
            if pd.notna(current_qty) and current_qty > 1e-9:
                est_annual_income_local = (
                    indicated_annual_dividend_rate_local * current_qty
                )

            # Dividend Yield on Cost (Local)
            if (
                pd.notna(current_qty)
                and current_qty > 1e-9
                and pd.notna(current_total_cost_local)
                and current_total_cost_local > 1e-9
            ):
                avg_cost_price_local = current_total_cost_local / current_qty
                if avg_cost_price_local > 1e-9:
                    div_yield_on_cost_pct_local = (
                        indicated_annual_dividend_rate_local / avg_cost_price_local
                    ) * 100.0

            # Dividend Yield on Current Value (Local)
            # Prefer computing from the indicated annual dividend rate and the
            # current price (both in local currency). This matches the
            # yield-on-cost methodology above and avoids Yahoo's ambiguous
            # `dividendYield` field, which inconsistently returns a fraction
            # (0.005) or a percentage number (0.5). The magnitude heuristic
            # used to disambiguate that field breaks for very low yields like
            # NVDA's ~0.03% (Yahoo returns 0.03, mistaken for a 3% fraction).
            if pd.notna(current_price_local) and current_price_local > 1e-9:
                div_yield_on_current_pct_local = (
                    indicated_annual_dividend_rate_local / current_price_local
                ) * 100.0
            elif (
                pd.notna(dividend_yield_on_current_direct)
                and dividend_yield_on_current_direct > 0
            ):
                div_yield_on_current_pct_local = (
                    dividend_yield_on_current_direct * 100.0
                    if dividend_yield_on_current_direct
                    < 1.0  # Yahoo usually returns fraction
                    else dividend_yield_on_current_direct
                )
        # --- END Unified Dividend Calculations ---

        # Convert dividend metrics to display currency
        div_yield_on_cost_pct_display = div_yield_on_cost_pct_local  # Yields are percentages, not currency dependent directly once calculated
        div_yield_on_current_pct_display = div_yield_on_current_pct_local

        est_annual_income_display = np.nan
        if pd.notna(est_annual_income_local) and pd.notna(fx_rate):
            est_annual_income_display = est_annual_income_local * fx_rate

        logging.debug(
            f"Symbol: {symbol}, Indicated Rate Local: {indicated_annual_dividend_rate_local}, Est Income Local: {est_annual_income_local}, Est Income Display: {est_annual_income_display}"
        )

        logging.debug(
            f"  Yield Cost Local: {div_yield_on_cost_pct_local}, Yield Current Local: {div_yield_on_current_pct_local}"
        )

        # --- End Dividend Calculations ---

        # --- ADDED: FX Gain/Loss Calculation ---
        fx_gain_loss_display_holding = np.nan
        fx_gain_loss_pct_holding = np.nan

        # --- Calculate Display Currency Values ---
        market_value_local = current_qty * current_price_local
        market_value_display = (
            market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        # --- Asset Change (Day Change) with FX and Transactions ---
        # Robust Logic: Market Gain = (Value Today) - (Value Yesterday) - (Net Flow Today)
        # This correctly isolates market movement from capital flows (buys/sells)
        day_change_value_display = np.nan

        # A holding whose local price move is unknown -- a fund priced from a
        # published NAV, a symbol with no feed priced off its last trade --
        # still moves in the display currency when FX does. Skipping the row
        # outright left its whole market value in the portfolio total, and in
        # the denominator of the day-change percentage, with none of the day's
        # currency move charged against it: a THB book worth $210k dropped
        # $1,915 of FX on a day the headline reported +$1,713. Price the FX leg
        # on its own and hold the price leg at "no move known".
        price_move_known = pd.notna(day_change_local)

        if (
            pd.notna(current_price_local)
            and pd.notna(fx_rate)
            and (price_move_known or local_currency != display_currency)
        ):
            # 1. Calculate Today's Net Flow and Quantity Change for this holding
            qty_change_today = 0.0
            net_flow_today_local = 0.0

            try:
                # Filter transactions for this specific holding on the report date
                # report_date is a datetime.date
                # MODIFIED: Include 'To Account' in filter for incoming transfers
                today = holding_rows.on_report_date()
                if today.empty:
                    tx_today = today
                elif "To Account" in transactions_df.columns:
                    tx_today = today[
                        (today["Symbol"] == symbol)
                        & (
                            (today["Account"] == account)
                            | (today["To Account"] == account)
                        )
                    ]
                else:
                    tx_today = today[
                        (today["Symbol"] == symbol) & (today["Account"] == account)
                    ]

                if not tx_today.empty:
                    for _, tx in tx_today.iterrows():
                        t_type = str(tx["Type"]).lower().strip()
                        t_qty = abs(float(tx["Quantity"]))
                        t_price = float(tx["Price/Share"])
                        t_comm = (
                            float(tx["Commission"])
                            if pd.notna(tx["Commission"])
                            else 0.0
                        )
                        t_acc = str(tx.get("Account", "")).strip()
                        t_to_acc = str(tx.get("To Account", "")).strip()

                        if t_type in ["buy", "deposit", "short sell"]:
                            # Buys/Deposits add to quantity (except short sell adds to the short position, which is negative qty)
                            qty_change_today += (
                                t_qty if t_type != "short sell" else -t_qty
                            )
                            net_flow_today_local += (t_qty * t_price) + t_comm
                        elif t_type in ["sell", "withdrawal", "buy to cover"]:
                            qty_change_today -= (
                                t_qty if t_type != "buy to cover" else -t_qty
                            )
                            net_flow_today_local -= (t_qty * t_price) - t_comm
                        elif t_type == "fees":
                            net_flow_today_local += t_comm
                        elif t_type == "transfer":
                            # Determine direction for this specific account
                            if t_to_acc == account:
                                # Incoming Transfer (Deposit-like)
                                qty_change_today += t_qty
                                net_flow_today_local += t_qty * t_price
                            elif t_acc == account:
                                # Outgoing Transfer (Withdrawal-like)
                                qty_change_today -= t_qty
                                net_flow_today_local -= t_qty * t_price
            except Exception as e_flow:
                logging.warning(
                    f"Error calculating intra-day flow for {symbol}/{account}: {e_flow}"
                )

            # 2. Get FX Previous Close
            fx_prev = np.nan
            if local_currency == display_currency:
                fx_prev = 1.0
            else:
                fx_prev = get_conversion_rate(
                    local_currency, display_currency, current_fx_prev_close_vs_usd
                )

            if pd.notna(fx_prev):
                # 3. Calculate Components
                starting_qty = current_qty - qty_change_today
                val_today = current_qty * current_price_local * fx_rate

                price_yesterday_local = (
                    current_price_local - day_change_local
                    if price_move_known
                    else current_price_local
                )
                val_yesterday = starting_qty * price_yesterday_local * fx_prev

                net_flow_today_display = net_flow_today_local * fx_rate

                # 4. Market Day Gain = Today Value - Yesterday Value - Net Flow
                day_change_value_display = (
                    val_today - val_yesterday - net_flow_today_display
                )

                # Measure the percentage on the same money as the figure above,
                # so the two agree wherever they are shown together. The
                # provider's percentage is the stock's move in its *own*
                # currency, which parts company with the display-currency
                # amount the moment FX moves — a holding can be up 0.13% in USD
                # and down ฿31,000 on the same day. Positions opened today have
                # no yesterday to measure against, so those keep the provider's.
                if abs(val_yesterday) > 1e-9:
                    day_change_pct = day_change_value_display / val_yesterday * 100.0
            else:
                # Fallback if FX prev missing: Ignore FX change, just use local change on starting qty
                starting_qty = current_qty - qty_change_today
                day_change_value_display = starting_qty * day_change_local * fx_rate
        current_price_display = (
            current_price_local * fx_rate
            if pd.notna(current_price_local) and pd.notna(fx_rate)
            else np.nan
        )
        cost_basis_display = np.nan
        avg_cost_price_display = np.nan
        unrealized_gain_display = np.nan
        unrealized_gain_pct = np.nan

        is_long = current_qty > 1e-9
        is_short = current_qty < -1e-9
        if is_long:
            cost_basis_display_local = max(0, current_total_cost_local)
            cost_basis_display = (
                cost_basis_display_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(cost_basis_display):
                avg_cost_price_display = (
                    (cost_basis_display / current_qty)
                    if abs(current_qty) > 1e-9
                    else np.nan
                )
                if pd.notna(market_value_display):
                    unrealized_gain_display = market_value_display - cost_basis_display
                    if abs(cost_basis_display) > 1e-9:
                        unrealized_gain_pct = (
                            unrealized_gain_display / cost_basis_display
                        ) * 100.0
                    elif abs(market_value_display) > 1e-9:
                        unrealized_gain_pct = np.inf
                    else:
                        unrealized_gain_pct = 0.0
        elif is_short:
            avg_cost_price_display = np.nan
            cost_basis_display = 0.0
            short_proceeds_display = (
                short_proceeds_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(market_value_display) and pd.notna(short_proceeds_display):
                current_cost_to_cover_display = abs(market_value_display)
                unrealized_gain_display = (
                    short_proceeds_display - current_cost_to_cover_display
                )
                if abs(short_proceeds_display) > 1e-9:
                    unrealized_gain_pct = (
                        unrealized_gain_display / short_proceeds_display
                    ) * 100.0
                elif abs(current_cost_to_cover_display) > 1e-9:
                    unrealized_gain_pct = -np.inf
                else:
                    unrealized_gain_pct = 0.0

        realized_gain_display = (
            realized_gain_display_from_holdings
            if pd.notna(realized_gain_display_from_holdings)
            else (realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan)
        )
        realized_gain_display = (
            realized_gain_display_from_holdings
            if pd.notna(realized_gain_display_from_holdings)
            else (realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan)
        )
        dividends_display = dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
        commissions_display = (
            commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        taxes_display = taxes_local * fx_rate if pd.notna(fx_rate) else np.nan
        total_cost_invested_display = (
            total_cost_invested_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        cumulative_investment_display = (
            cumulative_investment_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        total_buy_cost_display = (
            total_buy_cost_local * fx_rate if pd.notna(fx_rate) else np.nan
        )

        unrealized_gain_comp = (
            unrealized_gain_display if pd.notna(unrealized_gain_display) else 0.0
        )
        # Tax is treated like a cost subtraction in total gain (same as commissions).
        # If taxes_display is NaN (older snapshot or no tax data), treat as 0.
        taxes_comp = taxes_display if pd.notna(taxes_display) else 0.0
        total_gain_display = (
            (
                realized_gain_display
                + unrealized_gain_comp
                + dividends_display
                - commissions_display
                - taxes_comp
            )
            if all(
                pd.notna(v)
                for v in [realized_gain_display, dividends_display, commissions_display]
            )
            else np.nan
        )

        denominator_for_pct = total_buy_cost_display
        total_return_pct = np.nan
        if pd.notna(total_gain_display) and pd.notna(denominator_for_pct):
            if abs(denominator_for_pct) > 1e-9:
                total_return_pct = (total_gain_display / denominator_for_pct) * 100.0
            elif abs(total_gain_display) <= 1e-9:
                total_return_pct = 0.0

        stock_irr = np.nan
        try:
            market_value_local_for_irr = (
                abs(market_value_local)
                if abs(current_qty) > 1e-9 and pd.notna(market_value_local)
                else 0.0
            )
            cf_dates, cf_values = cash_flow_ledger.flows_for(
                symbol,
                account,
                market_value_local_for_irr,  # The final value of this specific holding
                is_transfer_a_flow=True,  # ADDED: Treat transfers as flows for IRR
                report_date=report_date,  # The date of the final value
            )
            if cf_dates and cf_values:
                # Ensure all dates are datetime.date objects to prevent TypeErrors
                # caused by string vs date comparison or calculate_irr validation.
                for i in range(len(cf_dates)):
                    d = cf_dates[i]
                    if isinstance(d, str):
                        try:
                            cf_dates[i] = datetime.strptime(d, "%Y-%m-%d").date()
                        except ValueError:
                            # Try ISO format or just keep as is if parse fails (will likely fail later)
                            pass
                    elif isinstance(d, datetime):
                        cf_dates[i] = d.date()

                # --- NEW: Suppress Annualized IRR for short duration (< 1 year) ---
                # Check actual holding duration including transfers (via Lots)
                # Removed flawed earliest_lot_date shift logic

                duration_days = (cf_dates[-1] - cf_dates[0]).days

                if duration_days >= 365:
                    stock_irr = calculate_irr(cf_dates, cf_values)
                else:
                    # Still suppress if true duration is < 1 year
                    # logging.debug(f"IRR suppressed for {symbol}/{account}: Duration {duration_days} days < 365 days.")
                    stock_irr = np.nan
        except Exception as e_irr:
            logging.warning(
                f"Warning: IRR calculation failed for {symbol}/{account}: {e_irr}",
                exc_info=True,
            )
            has_warnings = True
            stock_irr = np.nan

        # --- FX Gain/Loss Calculation (after all other local values are determined) ---
        fx_gain_loss_display_holding = np.nan  # Initialize
        fx_gain_loss_pct_holding = np.nan  # Initialize

        if local_currency == display_currency:
            fx_gain_loss_display_holding = 0.0
            fx_gain_loss_pct_holding = 0.0  # If no FX G/L, percentage is 0%
        elif (  # Original conditions, but now as elif
            abs(current_total_cost_local) > 1e-9  # Has a local cost basis
            and abs(current_total_cost_local) > 1e-9  # Has a local cost basis
            and pd.notna(cost_basis_display)  # Cost basis in display currency is valid
            and abs(cost_basis_display)
            > 1e-9  # Cost basis in display currency is significant
            and pd.notna(
                total_cost_display_historical_fx_val
            )  # Historical cost in display currency is valid
        ):
            try:
                # Cost at current FX: cost_basis_display (which is current_total_cost_local * current_fx_rate)
                # Cost at historical FX: total_cost_display_historical_fx_val
                fx_gain_loss_display_holding = (
                    cost_basis_display - total_cost_display_historical_fx_val
                )

                # Calculate percentage if the historical cost (denominator) is significant
                if (
                    pd.notna(fx_gain_loss_display_holding)
                    and abs(total_cost_display_historical_fx_val)
                    > 1e-9  # Denominator for % is historical cost
                ):
                    fx_gain_loss_pct_holding = (
                        fx_gain_loss_display_holding
                        / total_cost_display_historical_fx_val
                    ) * 100.0
                # If fx_gain_loss_display_holding is calculated as 0 (e.g. rates were identical or very close), pct should also be 0
                elif (
                    pd.notna(fx_gain_loss_display_holding)
                    and abs(fx_gain_loss_display_holding) < 1e-9
                ):
                    fx_gain_loss_pct_holding = 0.0
                # Else, fx_gain_loss_pct_holding remains np.nan if fx_gain_loss_display_holding is nan or historical cost is zero

            except ZeroDivisionError:
                logging.warning(
                    f"FX G/L Calc: ZeroDivisionError for {symbol}/{account}. current_total_cost_local: {current_total_cost_local}"
                )
            except Exception as e_fx_gl:
                logging.error(
                    f"Error calculating FX G/L for {symbol}/{account}: {e_fx_gl}"
                )
        # If none of the above conditions met (e.g. historical cost in display currency is NaN),
        # fx_gain_loss_display_holding and fx_gain_loss_pct_holding will remain np.nan as initialized.

        # --- END FX Gain/Loss Calculation ---

        irr_value_to_store = stock_irr * 100.0 if pd.notna(stock_irr) else np.nan
        if pd.isna(irr_value_to_store) and current_qty != 0:
            logging.debug(f"Debug: IRR is NaN for non-zero holding {symbol}/{account}")

        portfolio_summary_rows.append(
            {
                "Account": account,
                "Symbol": symbol,
                "Quantity": current_qty,
                f"Avg Cost ({display_currency})": avg_cost_price_display,
                f"Price ({display_currency})": current_price_display,
                f"Cost Basis ({display_currency})": cost_basis_display,
                f"Market Value ({display_currency})": market_value_display,
                f"Day Change ({display_currency})": day_change_value_display,
                "Day Change %": day_change_pct,  # Store % value directly
                f"Unreal. Gain ({display_currency})": unrealized_gain_display,
                "Unreal. Gain %": unrealized_gain_pct,
                f"Realized Gain ({display_currency})": realized_gain_display,
                f"Dividends ({display_currency})": dividends_display,
                f"Commissions ({display_currency})": commissions_display,
                f"Taxes ({display_currency})": taxes_display,
                f"Total Gain ({display_currency})": total_gain_display,
                f"Total Cost Invested ({display_currency})": total_cost_invested_display,
                "Total Return %": total_return_pct,
                f"Cumulative Investment ({display_currency})": cumulative_investment_display,
                f"Total Buy Cost ({display_currency})": total_buy_cost_display,
                "IRR (%)": irr_value_to_store,
                "Local Currency": local_currency,
                "Price Source": price_source,
                "Div. Yield (Cost) %": div_yield_on_cost_pct_display,
                "Div. Yield (Current) %": div_yield_on_current_pct_display,
                f"Est. Ann. Income ({display_currency})": est_annual_income_display,
                f"FX Gain/Loss ({display_currency})": fx_gain_loss_display_holding,
                "FX Gain/Loss %": fx_gain_loss_pct_holding,
                f"Tax ({display_currency})": (data.get("taxes_local", 0.0) * fx_rate)
                if pd.notna(fx_rate)
                else 0.0,
                "Name": stock_data.get("name", ""),  # Add Company Name
                "sparkline_7d": stock_data.get("sparkline_7d", []),
                "sparkline_1m": stock_data.get("sparkline_1m", []),
                "exchange": stock_data.get("exchange"),
                "fullExchangeName": stock_data.get("fullExchangeName"),
                "quoteType": stock_data.get("quoteType"),
                "Tags": tags_list,  # Added Tags
                "Market Value (Local)": market_value_local,  # ADDED
                "Price (Local)": current_price_local,  # ADDED
                "Avg Cost (Local)": (cost_basis_display_local / current_qty)
                if abs(current_qty) > 1e-9
                else np.nan,  # ADDED
                "fx_rate": fx_rate,  # ADDED: Multiplier from Local -> Display
            }
        )

        if symbol == "AAPL":
            logging.info(
                f"DEBUG_MARKET: AAPL Data -> Exchange: {stock_data.get('exchange')}, Full: {stock_data.get('fullExchangeName')}, MarketOverride: {data.get('Market')}"
            )
    # --- End Stock/ETF Loop ---

    # --- Loop 2: Process Cash Holdings (Per Account - No Aggregation) ---
    for currency, holding_list in cash_holdings_by_currency.items():
        if not holding_list:
            continue

        for h in holding_list:
            current_qty = h.get("qty", 0.0)

            # --- Skip cash rows with zero quantity ---
            if abs(current_qty) < STOCK_QUANTITY_CLOSE_TOLERANCE:
                continue

            account = h.get("account", "Unknown")

            realized_gain_local = h.get("realized_gain_local", 0.0)
            dividends_local = h.get("dividends_local", 0.0)
            commissions_local = h.get("commissions_local", 0.0)
            taxes_local = h.get("taxes_local", 0.0)
            current_total_cost_local = h.get("total_cost_local", 0.0)
            total_cost_invested_local = h.get("total_cost_invested_local", 0.0)
            cumulative_investment_local = h.get("cumulative_investment_local", 0.0)
            total_buy_cost_local = h.get("total_buy_cost_local", 0.0)
            total_cost_display_historical_fx_val = h.get(
                "total_cost_display_historical_fx", np.nan
            )

            local_currency = currency

            # --- Format cash symbol ---
            currency_symbol_for_display = get_currency_symbol_from_code(local_currency)
            cash_display_symbol = f"Cash ({currency_symbol_for_display})"

            price_source = "Internal (Cash)"
            current_price_local = 1.0
            day_change_local = 0.0
            day_change_pct = 0.0

            # --- Currency Conversion ---
            fx_rate = get_conversion_rate(
                local_currency, display_currency, current_fx_rates_vs_usd
            )

            if pd.isna(fx_rate):
                logging.error(
                    f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for cash in {account}."
                )
                has_errors = True
                fx_rate = np.nan

            est_annual_income_display = (
                0.0  # Will be populated in post-processing if rate > 0
            )

            market_value_display_for_yield = current_qty * (
                fx_rate if pd.notna(fx_rate) else 0.0
            )

            if market_value_display_for_yield > 1e-9:
                div_yield_on_current_pct_display = (
                    est_annual_income_display / market_value_display_for_yield
                ) * 100.0
            elif est_annual_income_display > 0:
                div_yield_on_current_pct_display = float("inf")
            else:
                div_yield_on_current_pct_display = 0.0

            # --- Calculate Display Currency Values ---
            market_value_local = current_qty * current_price_local
            market_value_display = (
                market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
            )

            # Asset Change logic
            day_change_value_display = 0.0
            day_change_pct = 0.0
            if (
                local_currency != display_currency
                and pd.notna(fx_rate)
                and pd.notna(current_qty)
            ):
                fx_prev = get_conversion_rate(
                    local_currency, display_currency, current_fx_prev_close_vs_usd
                )
                if pd.notna(fx_prev):
                    val_today = current_qty * 1.0 * fx_rate
                    val_yesterday = current_qty * 1.0 * fx_prev
                    day_change_value_display = val_today - val_yesterday
                    if abs(val_yesterday) > 1e-9:
                        day_change_pct = (
                            day_change_value_display / val_yesterday
                        ) * 100.0

            current_price_display = (
                current_price_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            cost_basis_display = market_value_display
            avg_cost_price_display = current_price_display

            unrealized_gain_display = 0.0
            unrealized_gain_pct = 0.0

            realized_gain_display = (
                realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            dividends_display = (
                dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            commissions_display = (
                commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            taxes_display = taxes_local * fx_rate if pd.notna(fx_rate) else np.nan
            total_cost_invested_display = (
                total_cost_invested_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            cumulative_investment_display = (
                cumulative_investment_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            total_buy_cost_display = (
                total_buy_cost_local * fx_rate if pd.notna(fx_rate) else np.nan
            )

            taxes_comp = taxes_display if pd.notna(taxes_display) else 0.0
            total_gain_display = np.nan
            if all(
                pd.notna(v)
                for v in [realized_gain_display, dividends_display, commissions_display]
            ):
                total_gain_display = (
                    realized_gain_display
                    + unrealized_gain_display
                    + dividends_display
                    - commissions_display
                    - taxes_comp
                )

            total_return_pct = np.nan
            if pd.notna(total_gain_display) and pd.notna(total_buy_cost_display):
                if abs(total_buy_cost_display) > 1e-9:
                    # Use Total Buy Cost (Cumulative Deposits) as the denominator for cash
                    # This provides a lifetime return on capital deposited.
                    total_return_pct = (
                        total_gain_display / total_buy_cost_display
                    ) * 100.0
                elif abs(total_gain_display) <= 1e-9:
                    total_return_pct = 0.0

            # A cash balance has no IRR of its own. Its external flows are only
            # deposits and withdrawals, while most of what passes through it is
            # sale proceeds and purchases inside the account, so an account
            # funded by transferred-in stock looks like $690 deposited and
            # $20,000 withdrawn: an "IRR" of 59,551%. The account's own return
            # (its MWR) is the meaningful figure; the cash row shows none.
            stock_irr = np.nan

            div_yield_on_cost_pct_display = np.nan

            fx_gain_loss_display_holding = np.nan
            fx_gain_loss_pct_holding = np.nan

            if local_currency == display_currency:
                fx_gain_loss_display_holding = 0.0
                fx_gain_loss_pct_holding = 0.0
            elif (
                pd.notna(cost_basis_display)
                and abs(cost_basis_display) > 1e-9
                and pd.notna(total_cost_display_historical_fx_val)
            ):
                try:
                    fx_gain_loss_display_holding = (
                        cost_basis_display - total_cost_display_historical_fx_val
                    )
                    if abs(total_cost_display_historical_fx_val) > 1e-9:
                        fx_gain_loss_pct_holding = (
                            fx_gain_loss_display_holding
                            / total_cost_display_historical_fx_val
                        ) * 100.0
                    elif abs(fx_gain_loss_display_holding) < 1e-9:
                        fx_gain_loss_pct_holding = 0.0
                except Exception as e_fx_gl_cash:
                    logging.error(
                        f"Error calculating FX G/L for cash {account}: {e_fx_gl_cash}"
                    )

            portfolio_summary_rows.append(
                {
                    "Account": account,
                    "Symbol": cash_display_symbol,
                    "Quantity": current_qty,
                    f"Avg Cost ({display_currency})": avg_cost_price_display,
                    f"Price ({display_currency})": current_price_display,
                    f"Cost Basis ({display_currency})": cost_basis_display,
                    f"Market Value ({display_currency})": market_value_display,
                    f"Day Change ({display_currency})": day_change_value_display,
                    "Day Change %": day_change_pct,
                    f"Unreal. Gain ({display_currency})": unrealized_gain_display,
                    "Unreal. Gain %": unrealized_gain_pct,
                    f"Realized Gain ({display_currency})": realized_gain_display,
                    f"Dividends ({display_currency})": dividends_display,
                    f"Commissions ({display_currency})": commissions_display,
                    f"Taxes ({display_currency})": taxes_display,
                    f"Total Gain ({display_currency})": total_gain_display,
                    f"Total Cost Invested ({display_currency})": total_cost_invested_display,
                    "Total Return %": total_return_pct,
                    f"Cumulative Investment ({display_currency})": cumulative_investment_display,
                    f"Total Buy Cost ({display_currency})": total_buy_cost_display,
                    "IRR (%)": stock_irr,
                    "Local Currency": local_currency,
                    "Price Source": price_source,
                    "Div. Yield (Current) %": np.nan,  # Will be updated in post-processing
                    f"Est. Ann. Income ({display_currency})": 0.0,  # Will be updated in post-processing
                    f"FX Gain/Loss ({display_currency})": fx_gain_loss_display_holding,
                    "FX Gain/Loss %": fx_gain_loss_pct_holding,
                    "Name": "Cash",
                    "Market Value (Local)": market_value_local,
                    "fx_rate": fx_rate,
                }
            )

    # --- Post-Process: Apply Cash Yield Interest (Aggregated by Account) ---
    if account_interest_rates:
        # 1. Group cash rows by account
        account_cash_map = defaultdict(list)
        for row in portfolio_summary_rows:
            if row.get("Symbol", "").startswith("Cash") or row.get("Symbol") == "$CASH":
                acc = row.get("Account")
                if acc:
                    account_cash_map[acc].append(row)

        # 2. Apply Threshold and Calculate Interest
        for acc_name, rows in account_cash_map.items():
            rate_pct = account_interest_rates.get(acc_name, 0.0)
            if rate_pct <= 0:
                continue

            thresh = interest_free_thresholds.get(acc_name, 0.0)

            # Extract fx_rate from the first cash row to convert threshold (assumed in USD) to display currency
            fx_rate = 1.0
            if rows and "fx_rate" in rows[0] and pd.notna(rows[0]["fx_rate"]):
                fx_rate = rows[0]["fx_rate"]

            thresh_display = thresh * fx_rate

            # Calculate Total Market Value (Display Currency) for this account's cash
            total_mv_display = sum(
                r.get(f"Market Value ({display_currency})", 0.0) for r in rows
            )

            # Calculate Effective Balance (Global Threshold Application)
            # Threshold is now properly converted to Display Currency
            effective_balance_display = max(0.0, total_mv_display - thresh_display)

            # Calculate Total Expected Annual Income
            total_income_display = effective_balance_display * (rate_pct / 100.0)

            # Distribute income back to rows (weighted by MV)
            if total_mv_display > 1e-9:
                for r in rows:
                    mv = r.get(f"Market Value ({display_currency})", 0.0)
                    weight = mv / total_mv_display
                    allocated_income = total_income_display * weight

                    r[f"Est. Ann. Income ({display_currency})"] = allocated_income
                    if mv > 1e-9:
                        r["Div. Yield (Current) %"] = (allocated_income / mv) * 100.0
            elif total_income_display > 0:
                # Edge case: Total MV is 0 or negative but somehow we have income? Unlikely with max(0, ...).
                pass
    # --- End Cash Loop ---

    # --- NEW: Calculate % of Total for all rows ---
    total_mv_display = sum(
        row.get(f"Market Value ({display_currency})", 0.0)
        for row in portfolio_summary_rows
        if pd.notna(row.get(f"Market Value ({display_currency})"))
    )

    for row in portfolio_summary_rows:
        mv = row.get(f"Market Value ({display_currency})", 0.0)
        if pd.notna(mv) and abs(total_mv_display) > 1e-9:
            row["pct_of_total"] = (mv / total_mv_display) * 100.0
        else:
            row["pct_of_total"] = 0.0
    # --- END NEW ---

    # --- NEW: Aggregate IRR Calculation (All Accounts View) ---
    # To fix the artificially low IRR seen when viewing "All Accounts", we calculate a
    # true aggregate IRR per symbol across all accounts, ignoring internal transfers.
    if portfolio_summary_rows:
        symbol_agg_mvs = defaultdict(float)
        # Sum local market value per symbol. (Assumes local currency is same across accounts for a given symbol)
        for r in portfolio_summary_rows:
            sym = r.get("Symbol")
            if sym and not str(sym).startswith("Cash"):
                symbol_agg_mvs[sym] += abs(float(r.get("Market Value (Local)", 0.0)))

        agg_irrs = {}
        for sym, total_mv_local in symbol_agg_mvs.items():
            try:
                # With account=None and is_transfer_a_flow=False, it gets true flows ignoring internal transfers
                cf_dates, cf_values = cash_flow_ledger.flows_for(
                    symbol=sym,
                    account=None,
                    final_market_value=float(total_mv_local),
                    is_transfer_a_flow=False,
                    report_date=report_date,
                )
                if cf_dates and cf_values:
                    for i in range(len(cf_dates)):
                        d = cf_dates[i]
                        if isinstance(d, str):
                            try:
                                cf_dates[i] = datetime.strptime(d, "%Y-%m-%d").date()
                            except ValueError:
                                pass
                        elif isinstance(d, datetime):
                            cf_dates[i] = d.date()

                    duration_days = (cf_dates[-1] - cf_dates[0]).days
                    if duration_days >= 365:
                        agg_irr = calculate_irr(cf_dates, cf_values)
                        if pd.notna(agg_irr):
                            agg_irrs[sym] = agg_irr * 100.0
            except Exception as e:
                logging.warning(f"Failed to calculate aggregate IRR for {sym}: {e}")

        # Inject aggregate IRR back into the rows
        for r in portfolio_summary_rows:
            sym = r.get("Symbol")
            if sym in agg_irrs:
                r["Aggregate IRR (%)"] = agg_irrs[sym]
                # We override the account's specific IRR with the Aggregate IRR.
                # Account-level IRR for transferred assets is fundamentally flawed without this.
                r["IRR (%)"] = agg_irrs[sym]
    # --- END AGGREGATE IRR ---

    # Drop rows belonging to accounts that are NOT in the user's selected
    # scope. See the docstring for the rationale — the upstream filter
    # deliberately keeps source-of-transfer transactions in the working set,
    # so this is the right place to hide the resulting phantom cash row.
    if include_accounts_norm is not None:
        kept_rows = []
        for r in portfolio_summary_rows:
            acc = str(r.get("Account") or "").strip().upper()
            if acc in include_accounts_norm:
                kept_rows.append(r)
        portfolio_summary_rows = kept_rows

    return (
        portfolio_summary_rows,
        dict(account_local_currency_map),
        has_errors,
        has_warnings,
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
    # Removed debug logging
    # ... (Function body remains unchanged, except it no longer references transactions_df) ...
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "mwr": np.nan,
            "total_return_pct": np.nan,
            "total_market_value_display": 0.0,
            "total_realized_gain_display": 0.0,
            "total_unrealized_gain_display": 0.0,
            "total_dividends_display": 0.0,
            "total_commissions_display": 0.0,
            "total_taxes_display": 0.0,
            "total_gain_display": 0.0,
            "total_cash_display": 0.0,
            "total_cost_invested_display": 0.0,
            "total_buy_cost_display": 0.0,
            "total_day_change_display": 0.0,
            "total_day_change_percent": np.nan,
        }
    )
    overall_summary_metrics = {}
    has_errors = False
    has_warnings = False

    logging.info("Calculating Account-Level & Overall Metrics...")
    if full_summary_df is None or full_summary_df.empty:
        logging.warning("Input 'full_summary_df' is empty or None.")
        overall_summary_metrics = {
            "market_value": 0.0,
            "cost_basis_held": 0.0,
            "unrealized_gain": 0.0,
            "realized_gain": 0.0,
            "dividends": 0.0,
            "commissions": 0.0,
            "taxes": 0.0,
            "total_gain": 0.0,
            "total_cost_invested": 0.0,
            "total_buy_cost": 0.0,
            "portfolio_mwr": np.nan,
            "day_change_display": 0.0,
            "day_change_percent": np.nan,
            "report_date": report_date.strftime("%Y-%m-%d"),
            "display_currency": display_currency,
            "cumulative_investment": 0.0,
            "total_return_pct": np.nan,
            "total_taxes": 0.0,
            "est_annual_income_display": 0.0,  # ADDED: Ensure key exists
        }
        return (overall_summary_metrics, dict(account_level_metrics), has_errors, True)

    required_cols = [
        "Account",
        "Symbol",
        "Quantity",
        f"Market Value ({display_currency})",
        f"Total Gain ({display_currency})",
        f"Total Buy Cost ({display_currency})",
    ]
    missing_cols = [col for col in required_cols if col not in full_summary_df.columns]
    if missing_cols:
        logging.warning(
            f"Warning: Aggregate calculation input missing required columns: {missing_cols}. Totals may be inaccurate."
        )
        has_warnings = True

    unique_accounts_in_summary = full_summary_df["Account"].unique()
    for account in unique_accounts_in_summary:
        try:
            account_full_df = full_summary_df[full_summary_df["Account"] == account]
            metrics_entry = account_level_metrics[account]
            metrics_entry["mwr"] = np.nan
            cols_to_sum_display = [
                f"Market Value ({display_currency})",
                f"Realized Gain ({display_currency})",
                f"Unreal. Gain ({display_currency})",
                f"Dividends ({display_currency})",
                f"Commissions ({display_currency})",
                f"Total Gain ({display_currency})",
                f"Total Cost Invested ({display_currency})",
                f"Total Buy Cost ({display_currency})",
                f"Day Change ({display_currency})",
                f"FX Gain/Loss ({display_currency})",  # Add FX G/L for aggregation
                f"Tax ({display_currency})",
            ]
            for col in cols_to_sum_display:
                if col not in account_full_df.columns:
                    logging.warning(
                        f"Warn: Col '{col}' missing for acc '{account}' agg."
                    )
                    has_warnings = True

            # --- ADDED: Account-level FX Gain/Loss ---
            acc_fx_gain_loss_display = safe_sum(
                account_full_df, f"FX Gain/Loss ({display_currency})"
            )
            metrics_entry["fx_gain_loss_display"] = acc_fx_gain_loss_display

            acc_cost_basis_display_for_fx_pct = safe_sum(
                account_full_df, f"Cost Basis ({display_currency})"
            )
            metrics_entry["fx_gain_loss_pct"] = (
                (acc_fx_gain_loss_display / acc_cost_basis_display_for_fx_pct) * 100
                if abs(acc_cost_basis_display_for_fx_pct) > 1e-9
                else np.nan
            )
            # --- END ADDED ---

            # FIX: EXCLUDE FX Gain/Loss from Market Value (It is already intrinsic in Price * FX conversion)
            metrics_entry["total_market_value_display"] = safe_sum(
                account_full_df, f"Market Value ({display_currency})"
            )

            metrics_entry["total_realized_gain_display"] = safe_sum(
                account_full_df, f"Realized Gain ({display_currency})"
            )
            # FIX: EXCLUDE FX Gain/Loss from Unrealized Gain (Intrinsic)
            metrics_entry["total_unrealized_gain_display"] = safe_sum(
                account_full_df, f"Unreal. Gain ({display_currency})"
            )

            metrics_entry["total_dividends_display"] = safe_sum(
                account_full_df, f"Dividends ({display_currency})"
            )
            metrics_entry["total_taxes_display"] = (
                safe_sum(account_full_df, f"Taxes ({display_currency})")
                if f"Taxes ({display_currency})" in account_full_df.columns
                else 0.0
            )
            metrics_entry["total_commissions_display"] = safe_sum(
                account_full_df, f"Commissions ({display_currency})"
            )
            # FIX: EXCLUDE FX Gain/Loss from Total Gain (Intrinsic)
            metrics_entry["total_gain_display"] = safe_sum(
                account_full_df, f"Total Gain ({display_currency})"
            )

            metrics_entry["total_taxes_display"] = safe_sum(
                account_full_df, f"Tax ({display_currency})"
            )

            # Sum market value of rows identified as Cash
            if "Name" in account_full_df.columns:
                acc_cash_rows = account_full_df[account_full_df["Name"] == "Cash"]
                metrics_entry["total_cash_display"] = safe_sum(
                    acc_cash_rows, f"Market Value ({display_currency})"
                )
            else:
                metrics_entry["total_cash_display"] = 0.0
            metrics_entry["total_cost_invested_display"] = safe_sum(
                account_full_df, f"Total Cost Invested ({display_currency})"
            )
            acc_total_buy_cost_display = safe_sum(
                account_full_df, f"Total Buy Cost ({display_currency})"
            )
            metrics_entry["total_buy_cost_display"] = acc_total_buy_cost_display
            acc_total_gain = metrics_entry["total_gain_display"]
            acc_denominator = acc_total_buy_cost_display
            acc_total_return_pct = np.nan
            if pd.notna(acc_total_gain) and pd.notna(acc_denominator):
                if abs(acc_denominator) > 1e-9:
                    acc_total_return_pct = (acc_total_gain / acc_denominator) * 100.0
                elif abs(acc_total_gain) <= 1e-9:
                    acc_total_return_pct = 0.0
            metrics_entry["total_return_pct"] = acc_total_return_pct
            acc_total_day_change_display = safe_sum(
                account_full_df, f"Day Change ({display_currency})"
            )
            # --- DEBUG LOGGING ---
            # Log the top contributors to Day Change to trace fluctuations
            if abs(acc_total_day_change_display) > 0:
                day_change_breakdown = account_full_df[
                    [
                        "Symbol",
                        f"Day Change ({display_currency})",
                        f"Market Value ({display_currency})",
                    ]
                ].copy()
                day_change_breakdown = day_change_breakdown.sort_values(
                    by=f"Day Change ({display_currency})", key=abs, ascending=False
                ).head(5)  # Top 5 movers

                logging.info(
                    f"Day Change Breakdown for {account}: Total={acc_total_day_change_display}"
                )
                for _, row in day_change_breakdown.iterrows():
                    logging.info(
                        f"  > {row['Symbol']}: DayChg={row[f'Day Change ({display_currency})']}, "
                        f"MV={row[f'Market Value ({display_currency})']}"
                    )
            # ---------------------
            metrics_entry["total_day_change_display"] = acc_total_day_change_display
            acc_current_mv_display = metrics_entry["total_market_value_display"]
            acc_prev_close_mv_display = np.nan
            if pd.notna(acc_current_mv_display) and pd.notna(
                acc_total_day_change_display
            ):
                acc_prev_close_mv_display = (
                    acc_current_mv_display - acc_total_day_change_display
                )
            metrics_entry["total_day_change_percent"] = np.nan
            if pd.notna(acc_total_day_change_display) and pd.notna(
                acc_prev_close_mv_display
            ):
                if abs(acc_prev_close_mv_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        acc_total_day_change_display / acc_prev_close_mv_display
                    ) * 100.0
                elif abs(acc_total_day_change_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        np.inf if acc_total_day_change_display > 0 else -np.inf
                    )
                elif abs(acc_total_day_change_display) < 1e-9:
                    metrics_entry["total_day_change_percent"] = 0.0

        except Exception:
            logging.exception(f"Error aggregating metrics for account '{account}'")
            has_warnings = True

    # --- START FIX: Filter for Overall Summary ---
    df_for_overall_summary = full_summary_df

    # Determine if the filter is for "All Accounts"
    is_all_accounts_selected = False
    if not include_accounts:  # None or empty list
        is_all_accounts_selected = True
    elif all_available_accounts and set(include_accounts) == set(
        all_available_accounts
    ):
        is_all_accounts_selected = True

    if not is_all_accounts_selected:
        # Filter the DataFrame to *only* the accounts the user selected
        logging.debug(
            f"Aggregating overall metrics for selected accounts: {include_accounts}"
        )
        if "Account" in df_for_overall_summary.columns:
            # Keep rows that match the selected accounts
            account_mask = full_summary_df["Account"].isin(include_accounts)

            # --- MODIFIED: Also keep the special aggregate cash row ---
            # When filtering for specific accounts, we must still include the aggregated cash
            # row to ensure the 'Overall' portfolio metrics (like total market value) are correct.
            cash_mask = full_summary_df["Account"] == _AGGREGATE_CASH_ACCOUNT_NAME_

            # Combine the masks to keep selected accounts AND the cash row.
            df_for_overall_summary = full_summary_df[account_mask | cash_mask]
            # --- END MODIFICATION ---
        else:
            logging.warning("Cannot filter overall metrics: 'Account' column missing.")
    # --- END FIX ---

    # Note: The original code had a second part to the filter for the "All Accounts" case.
    # That logic is now correctly handled by the `show_closed_positions` filter in the main summary function,
    # which removes zero-quantity transfer source rows before they even get here.
    # Therefore, only the explicit account filter is needed for the overall summary calculation.

    mkt_val_col = f"Market Value ({display_currency})"
    total_gain_col = f"Total Gain ({display_currency})"
    total_buy_cost_col = f"Total Buy Cost ({display_currency})"
    cost_basis_col = f"Cost Basis ({display_currency})"
    unreal_gain_col = f"Unreal. Gain ({display_currency})"
    real_gain_col = f"Realized Gain ({display_currency})"
    divs_col = f"Dividends ({display_currency})"
    comm_col = f"Commissions ({display_currency})"
    taxes_col = f"Taxes ({display_currency})"
    cost_invest_col = f"Total Cost Invested ({display_currency})"
    cum_invest_col = f"Cumulative Investment ({display_currency})"
    day_change_col = f"Day Change ({display_currency})"
    fx_gain_loss_col = f"FX Gain/Loss ({display_currency})"  # For overall sum
    cols_to_check = [
        mkt_val_col,
        total_gain_col,
        total_buy_cost_col,
        cost_basis_col,
        unreal_gain_col,
        real_gain_col,
        divs_col,
        comm_col,
        cost_invest_col,
        cum_invest_col,
        day_change_col,
        fx_gain_loss_col,
        f"Tax ({display_currency})",
    ]
    for col in cols_to_check:
        if col not in df_for_overall_summary.columns:
            logging.warning(f"Warning: Column '{col}' missing for overall aggregation.")
            has_warnings = True

    # --- ADDED: Overall FX Gain/Loss ---
    overall_fx_gain_loss_display = safe_sum(df_for_overall_summary, fx_gain_loss_col)
    # --- END ADDED ---

    # FIX: EXCLUDE FX Gain/Loss from Overall Market Value (Intrinsic)
    raw_sum_mkt = safe_sum(df_for_overall_summary, mkt_val_col)
    overall_market_value_display = raw_sum_mkt

    held_mask = pd.Series(False, index=df_for_overall_summary.index)
    if (
        "Quantity" in df_for_overall_summary.columns
    ):  # Symbol check is implicit as cash is now a holding
        held_mask = df_for_overall_summary["Quantity"].abs() > 1e-9
    overall_cost_basis_display = (
        safe_sum(df_for_overall_summary.loc[held_mask], cost_basis_col)
        if held_mask.any()
        else 0.0
    )
    # FIX: EXCLUDE FX Gain/Loss from Overall Unrealized Gain (Intrinsic)
    overall_unrealized_gain_display = safe_sum(df_for_overall_summary, unreal_gain_col)

    overall_realized_gain_display_agg = safe_sum(df_for_overall_summary, real_gain_col)
    overall_dividends_display_agg = safe_sum(df_for_overall_summary, divs_col)
    overall_commissions_display_agg = safe_sum(df_for_overall_summary, comm_col)
    overall_taxes_display_agg = (
        safe_sum(df_for_overall_summary, taxes_col)
        if taxes_col in df_for_overall_summary.columns
        else 0.0
    )

    # FIX: EXCLUDE FX Gain/Loss from Overall Total Gain (Intrinsic)
    overall_total_gain_display = safe_sum(df_for_overall_summary, total_gain_col)

    overall_total_taxes_display = safe_sum(
        df_for_overall_summary, f"Tax ({display_currency})"
    )

    overall_total_cost_invested_display = safe_sum(
        df_for_overall_summary, cost_invest_col
    )
    overall_cumulative_investment_display = safe_sum(
        df_for_overall_summary, cum_invest_col
    )
    overall_total_buy_cost_display = safe_sum(
        df_for_overall_summary, total_buy_cost_col
    )
    overall_day_change_display = safe_sum(df_for_overall_summary, day_change_col)
    overall_prev_close_mv_display = np.nan

    # --- ADDED: Overall Estimated Annual Income ---
    est_ann_income_col = f"Est. Ann. Income ({display_currency})"
    overall_est_annual_income_display = safe_sum(
        df_for_overall_summary, est_ann_income_col
    )
    # --- END ADDED ---

    # --- ADDED: Overall Cash Balance ---
    overall_cash_balance_display = 0.0
    if "Name" in df_for_overall_summary.columns:
        overall_cash_rows = df_for_overall_summary[
            df_for_overall_summary["Name"] == "Cash"
        ]
        overall_cash_balance_display = safe_sum(overall_cash_rows, mkt_val_col)
    # --- END ADDED ---

    if pd.notna(overall_market_value_display) and pd.notna(overall_day_change_display):
        overall_prev_close_mv_display = (
            overall_market_value_display - overall_day_change_display
        )
    overall_day_change_percent = np.nan
    if pd.notna(overall_day_change_display) and pd.notna(overall_prev_close_mv_display):
        if abs(overall_prev_close_mv_display) > 1e-9:
            overall_day_change_percent = (
                overall_day_change_display / overall_prev_close_mv_display
            ) * 100.0
        elif abs(overall_day_change_display) > 1e-9:
            overall_day_change_percent = (
                np.inf if overall_day_change_display > 0 else -np.inf
            )
        elif abs(overall_day_change_display) < 1e-9:
            overall_day_change_percent = 0.0

    overall_total_return_pct = np.nan
    # IMPROVEMENT: Use overall_total_buy_cost_display (Cumulative Deposits) as the denominator
    # for the overall portfolio return to prevent extreme percentages when the remaining balance is small.
    overall_denominator = overall_total_buy_cost_display

    if pd.notna(overall_total_gain_display) and pd.notna(overall_denominator):
        if abs(overall_denominator) > 1e-9:
            overall_total_return_pct = (
                overall_total_gain_display / overall_denominator
            ) * 100.0
        elif abs(overall_total_gain_display) <= 1e-9:
            overall_total_return_pct = 0.0

    critical_overall_metrics = [
        overall_market_value_display,
        overall_total_gain_display,
    ]
    if any(pd.isna(m) for m in critical_overall_metrics):
        logging.warning("Warning: Critical overall metrics calculated as NaN.")
        has_warnings = True

    # --- ADDED: Overall FX Gain/Loss Percentage ---
    overall_fx_gain_loss_pct = (
        (overall_fx_gain_loss_display / overall_cost_basis_display) * 100
        if abs(overall_cost_basis_display) > 1e-9
        else np.nan
    )
    # --- END ADDED ---

    # --- ADDED: Overall Forward Dividend Yield Percentage ---
    overall_dividend_yield_pct = (
        (overall_est_annual_income_display / overall_market_value_display) * 100
        if abs(overall_market_value_display) > 1e-9
        else 0.0
    )
    # --- END ADDED ---

    # --- ADDED: Overall MWR (IRR) Calculation ---
    overall_mwr = np.nan

    # We need to pass the *filtered* transactions relevant to these accounts.
    # df_for_overall_summary is 'Summary' data, not transactions.
    # 'transactions_df' is passed into this function.
    if transactions_df is not None and not transactions_df.empty:
        # Filter transactions for the selected accounts (if any)
        # Note: If no accounts selected (empty list), we include all.
        # Filter transactions for the selected accounts (if any)
        # Note: If no accounts selected (empty list), we include all.
        tx_for_mwr = transactions_df
        if not is_all_accounts_selected:
            if "Account" in transactions_df.columns:
                # FIX: Must include transactions where Account is in list OR To Account is in list (Transfer In)
                # Case-insensitive matching is CRITICAL

                include_accounts_upper = [
                    str(a).strip().upper() for a in include_accounts
                ]

                # Check Account col
                acc_series = (
                    transactions_df["Account"].astype(str).str.strip().str.upper()
                )
                mask = acc_series.isin(include_accounts_upper)

                if "To Account" in transactions_df.columns:
                    to_acc_series = (
                        transactions_df["To Account"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                    )
                    mask = mask | to_acc_series.isin(include_accounts_upper)

                tx_for_mwr = transactions_df[mask]

        # Calculate Cash Flows (portfolio-level basis: buys/sells/divs are internal, not flows)
        mwr_dates, mwr_flows = get_cash_flows_for_mwr(
            account_transactions=tx_for_mwr,
            final_account_market_value=overall_market_value_display,
            end_date=report_date,
            target_currency=display_currency,
            fx_rates=fx_rates,
            historical_fx_rates=historical_fx_rates,  # ADDED: Historical FX Data
            display_currency=display_currency,
            include_accounts=(
                include_accounts if include_accounts else all_available_accounts
            ),  # PASS SCOPE for Transfer logic
            flow_basis="portfolio",
        )

        if mwr_dates and mwr_flows:
            overall_mwr = calculate_irr(mwr_dates, mwr_flows)
            if pd.notna(overall_mwr):
                overall_mwr = overall_mwr * 100.0  # Convert to percentage
            else:
                logging.debug(
                    f"MWR Calc Failed (Result NaN). Dates: {len(mwr_dates)}, Flows: {len(mwr_flows)}"
                )
        else:
            logging.debug(
                f"MWR Prep Failed (No flows). Market Val: {overall_market_value_display}, TX Count: {len(tx_for_mwr) if not tx_for_mwr.empty else 0}"
            )

    # --- END ADDED ---

    # --- ADDED: Account-Level MWR (IRR) Calculation ---
    if (
        transactions_df is not None
        and not transactions_df.empty
        and "Account" in transactions_df.columns
    ):
        # We need to calculate MWR for each account in account_level_metrics
        # Note: account_level_metrics keys are account names.
        # The normalised account columns are the same for every account:
        # built once (a failure is kept and raised in each account's handler).
        normalised: Dict[str, Any] = {}

        def upper_column(name: str) -> pd.Series:
            if name not in normalised:
                try:
                    normalised[name] = (
                        True,
                        transactions_df[name].astype(str).str.strip().str.upper(),
                    )
                except Exception as exc:
                    normalised[name] = (False, exc)
            ok, value = normalised[name]
            if not ok:
                raise value
            return value

        for account in account_level_metrics:
            try:
                # Filter transactions for this specific account
                # Match account name exactly (including transfers IN where this account is "To Account")
                # Case-insensitive match
                account_upper = str(account).strip().upper()

                acc_series_sub = upper_column("Account")
                to_acc_series_sub = pd.Series(False, index=transactions_df.index)
                if "To Account" in transactions_df.columns:
                    to_acc_series_sub = upper_column("To Account") == account_upper

                account_tx = transactions_df[
                    (acc_series_sub == account_upper) | to_acc_series_sub
                ]

                if account_tx.empty:
                    continue

                # Get current market value for this account from the metrics we just built
                # account_level_metrics is a dict of dicts
                account_mv = account_level_metrics[account].get(
                    "total_market_value_display", 0.0
                )

                # Calculate Cash Flows using the helper (portfolio basis for per-account MWR)
                mwr_dates, mwr_flows = get_cash_flows_for_mwr(
                    account_transactions=account_tx,
                    final_account_market_value=account_mv,
                    end_date=report_date,
                    target_currency=display_currency,
                    fx_rates=fx_rates,
                    historical_fx_rates=historical_fx_rates,
                    display_currency=display_currency,
                    include_accounts=[account],  # PASS SINGLE ACCOUNT SCOPE
                    flow_basis="portfolio",
                )

                if mwr_dates and mwr_flows:
                    account_mwr = calculate_irr(mwr_dates, mwr_flows)
                    if pd.notna(account_mwr):
                        # Update the metric for this account
                        # The dashboard expects 'portfolio_mwr' or 'mwr'
                        account_level_metrics[account]["mwr"] = account_mwr * 100.0
                        account_level_metrics[account]["portfolio_mwr"] = (
                            account_mwr * 100.0
                        )  # Map for safety
                    else:
                        # Calculation resulted in NaN (no solution or invalid data)
                        pass
            except Exception as e_acct_mwr:
                logging.warning(
                    f"Failed to calculate MWR for account '{account}': {e_acct_mwr}"
                )
    # --- END ACCOUNT MWR ---

    overall_summary_metrics = {
        "market_value": overall_market_value_display,
        "cost_basis_held": overall_cost_basis_display,
        "unrealized_gain": overall_unrealized_gain_display,
        "realized_gain": overall_realized_gain_display_agg,
        "dividends": overall_dividends_display_agg,
        "commissions": overall_commissions_display_agg,
        "taxes": overall_taxes_display_agg,
        "total_gain": overall_total_gain_display,
        "total_cost_invested": overall_total_cost_invested_display,
        "total_buy_cost": overall_total_buy_cost_display,
        "portfolio_mwr": overall_mwr,  # Using the calculated value
        "day_change_display": overall_day_change_display,
        "day_change_percent": overall_day_change_percent,
        "report_date": report_date.strftime("%Y-%m-%d"),
        "display_currency": display_currency,
        "cumulative_investment": overall_cumulative_investment_display,
        "total_return_pct": overall_total_return_pct,
        "fx_gain_loss_display": overall_fx_gain_loss_display,  # ADDED
        "fx_gain_loss_pct": overall_fx_gain_loss_pct,  # ADDED
        "est_annual_income_display": overall_est_annual_income_display,  # ADDED
        "dividend_yield_pct": overall_dividend_yield_pct,  # ADDED
        "cash_balance": overall_cash_balance_display,  # ADDED
        "total_taxes": overall_total_taxes_display,  # ADDED
    }
    logging.debug(
        f"--- Finished Aggregating Metrics. Overall Market Value: {overall_market_value_display} ---"
    )

    return (
        overall_summary_metrics,
        dict(account_level_metrics),
        has_errors,
        has_warnings,
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


# Event kinds for investa_core.fifo_lots.
_FIFO_TRANSFER, _FIFO_SPLIT, _FIFO_OPEN, _FIFO_REDUCE_BASIS, _FIFO_SELL = range(5)


def _fifo_float(value: Any) -> float:
    """A rate or price for the Rust lot book: None and NaN both mean missing."""
    return np.nan if value is None else float(value)


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

    Each row is read and priced here, then turned into one lot event; the lot
    bookkeeping (transfers, global splits, spin-off legs, buys and FIFO sells)
    runs in investa_core.fifo_lots.
    """
    output_columns = [
        "Date",
        "Symbol",
        "Account",
        "Type",
        "Quantity",
        "Avg Sale Price (Local)",
        "Total Proceeds (Local)",
        "Total Cost Basis (Local)",
        "Realized Gain (Local)",
        "Sale/Cover FX Rate",
        "Total Proceeds (Display)",
        "Total Cost Basis (Display)",
        "Realized Gain (Display)",
        "LocalCurrency",
        "original_tx_id",
    ]

    # (symbol, account) -> holding id; symbols get ids of their own for splits.
    holding_ids: Dict[Tuple[str, str], int] = {}
    holding_keys: List[Tuple[str, str]] = []
    symbol_ids: Dict[str, int] = {}
    symbol_of: List[int] = []

    def holding(key: Tuple[str, str]) -> int:
        h = holding_ids.get(key)
        if h is None:
            h = holding_ids[key] = len(holding_keys)
            holding_keys.append(key)
            symbol_of.append(symbol_ids.setdefault(key[0], len(symbol_ids)))
        return h

    kinds: List[int] = []
    col_a: List[int] = []
    col_b: List[int] = []
    col_x: List[float] = []
    col_y: List[float] = []
    col_z: List[float] = []
    col_w: List[float] = []
    days: List[int] = []
    # Per event: the lot's (purchase date, FX, id) objects for an open, the
    # partial gains record for a sell.
    context: List[Any] = []

    def emit(kind, a=0, b=0, x=np.nan, y=np.nan, z=np.nan, w=np.nan, day=0, ctx=None):
        kinds.append(kind)
        col_a.append(a)
        col_b.append(b)
        col_x.append(float(x))
        col_y.append(float(y))
        col_z.append(float(z))
        col_w.append(float(w))
        days.append(day)
        context.append(ctx)

    def open_lot(key, qty, cps, tx_date, fx, tx_id):
        emit(
            _FIFO_OPEN,
            a=holding(key),
            x=qty,
            y=cps,
            z=_fifo_float(fx),
            day=tx_date.toordinal(),
            ctx=(tx_date, fx, tx_id),
        )

    # The historical rate depends only on the currency and the day here, and a
    # ledger repeats both (1,495 lookups, 326 distinct cross-currency ones on a
    # real book), so each is asked once per call.
    fx_memo: Dict[Tuple[str, date], Any] = {}

    def fx_on(local_curr: str, tx_date: date) -> Any:
        key = (local_curr, tx_date)
        if key not in fx_memo:
            fx_memo[key] = get_historical_rate_via_usd_bridge(
                local_curr, display_currency, tx_date, historical_fx_yf
            )
        return fx_memo[key]

    # Track splits already applied to avoid double-processing (symbol, date, ratio)
    processed_splits: Set[Tuple[str, date, float]] = set()

    # Columns as iterrows() would hand them out, without building a row Series
    # per transaction (row.get(col) is None for a missing column).
    n_rows = len(transactions_df)

    def column(name, missing=None):
        if name in transactions_df.columns:
            return transactions_df[name].tolist()
        return [missing] * n_rows

    required = ["Date", "Symbol", "Account", "Type"]
    if any(c not in transactions_df.columns for c in required):
        for c in required:
            if c not in transactions_df.columns:
                logging.warning(f"Error parsing basic data for FIFO row: {c!r}")
                break
        return pd.DataFrame(columns=output_columns), {}

    rows = zip(
        column("Date"),
        column("Symbol"),
        column("Account"),
        column("Type"),
        column("Local Currency"),
        column("Quantity"),
        column("Price/Share"),
        column("Commission"),
        column("Split Ratio"),
        column("original_index"),
        column("To Account", ""),
        column("Total Amount"),
    )
    for (
        tx_date_dt,
        symbol_raw,
        account_raw,
        type_raw,
        local_curr_raw,
        qty_raw,
        price_local_raw,
        commission_local_raw,
        split_ratio_raw,
        original_tx_id_current_row,
        to_account_raw,
        total_amount_raw,
    ) in rows:
        try:
            if pd.isna(tx_date_dt):
                continue
            tx_date = tx_date_dt.date()

            symbol = str(symbol_raw).upper().strip()
            account = str(account_raw).upper().strip()
            tx_type = str(type_raw).lower().strip()
            local_curr = str(
                local_curr_raw
                if pd.notna(local_curr_raw) and str(local_curr_raw).strip()
                else default_currency
            ).upper()

            if not local_curr or len(local_curr) != 3:
                local_curr = default_currency.upper()

            qty = pd.to_numeric(qty_raw, errors="coerce")
            price_local = pd.to_numeric(price_local_raw, errors="coerce")
            commission_local = (
                0.0
                if pd.isna(commission_local_raw)
                else float(pd.to_numeric(commission_local_raw, errors="coerce"))
            )
            split_ratio = pd.to_numeric(split_ratio_raw, errors="coerce")

        except Exception as e_parse:
            logging.warning(f"Error parsing basic data for FIFO row: {e_parse}")
            continue

        holding_key = (symbol, account)

        if symbol == CASH_SYMBOL_CSV or tx_type in [
            "deposit",
            "withdrawal",
            "fees",
            "dividend",
        ]:
            continue

        # --- Handle Transfers ---
        if tx_type == "transfer":
            to_account = str(to_account_raw).upper().strip()
            if not to_account:
                continue
            emit(
                _FIFO_TRANSFER,
                a=holding(holding_key),
                b=holding((symbol, to_account)),
                x=qty,
            )
            continue

        if tx_type in ["split", "stock split"]:
            if pd.notna(split_ratio) and split_ratio > 0:
                split_event = (symbol, tx_date, float(split_ratio))
                if split_event not in processed_splits:
                    # Global split: apply to all accounts holding this symbol
                    emit(
                        _FIFO_SPLIT,
                        a=symbol_ids.setdefault(symbol, len(symbol_ids)),
                        x=split_ratio,
                    )
                    processed_splits.add(split_event)
            continue

        # --- Spin-off (cash-neutral basis reallocation, two legs by qty) ---
        if tx_type in ["spin off", "spin-off", "spinoff"]:
            total_amt_raw = pd.to_numeric(total_amount_raw, errors="coerce")
            total_amt = abs(float(total_amt_raw)) if pd.notna(total_amt_raw) else 0.0
            if pd.notna(qty) and qty > 1e-9:
                # Child receipt: open a lot at the allocated per-share basis.
                cps = total_amt / qty if qty > 1e-9 else 0.0
                fx_child = fx_on(local_curr, tx_date)
                open_lot(
                    holding_key, qty, cps, tx_date, fx_child, original_tx_id_current_row
                )
            else:
                # Parent basis reduction: the remaining lots' total basis drops
                # by `total_amt` (clamped >= 0).
                emit(_FIFO_REDUCE_BASIS, a=holding(holding_key), x=total_amt)
            continue

        if pd.isna(qty) or qty <= 1e-9:
            continue

        fx_rate_to_display_current_tx = fx_on(local_curr, tx_date)

        if tx_type == "buy":
            if pd.isna(price_local) or price_local <= 1e-9:
                total_amt_raw = pd.to_numeric(total_amount_raw, errors="coerce")
                if (
                    pd.notna(total_amt_raw)
                    and abs(float(total_amt_raw)) > 1e-9
                    and qty > 1e-9
                ):
                    price_local = abs(float(total_amt_raw)) / qty
                else:
                    continue

            cost_per_share_local_net = ((qty * price_local) + commission_local) / qty
            open_lot(
                holding_key,
                qty,
                cost_per_share_local_net,
                tx_date,
                fx_rate_to_display_current_tx,
                original_tx_id_current_row,
            )

        elif tx_type == "sell":
            if pd.isna(price_local):
                total_amt_raw = pd.to_numeric(total_amount_raw, errors="coerce")
                if pd.notna(total_amt_raw) and qty > 1e-9:
                    price_local = abs(float(total_amt_raw)) / qty
                else:
                    price_local = 0.0
            price_local = max(0.0, float(price_local))
            emit(
                _FIFO_SELL,
                a=holding(holding_key),
                x=qty,
                y=price_local,
                z=commission_local,
                w=_fifo_float(fx_rate_to_display_current_tx),
                ctx={
                    "Date": tx_date,
                    "Symbol": symbol,
                    "Account": account,
                    "Type": "Sale Long",
                    "Quantity": qty,
                    "Avg Sale Price (Local)": price_local,
                    "Sale/Cover FX Rate": fx_rate_to_display_current_tx,
                    "LocalCurrency": local_curr,
                    "original_tx_id": original_tx_id_current_row,
                },
            )

    # Lots of one purchase date stay in transaction-id order after a transfer;
    # the Rust book compares ids by rank.
    open_events = [i for i, k in enumerate(kinds) if k == _FIFO_OPEN]
    id_rank = {v: r for r, v in enumerate(sorted({context[i][2] for i in open_events}))}
    ranks = [0] * len(kinds)
    for i in open_events:
        ranks[i] = id_rank[context[i][2]]

    sales, (keys, offsets, lot_qty, lot_cps, lot_source) = investa_core.fifo_lots(
        kinds,
        col_a,
        col_b,
        col_x,
        col_y,
        col_z,
        col_w,
        days,
        ranks,
        symbol_of,
        float(stock_quantity_close_tolerance),
    )

    open_lots: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for k, h in enumerate(keys):
        lots = []
        for j in range(offsets[k], offsets[k + 1]):
            purchase_date, fx, tx_id = context[lot_source[j]]
            lots.append(
                {
                    "qty": lot_qty[j],
                    "cost_per_share_local_net": lot_cps[j],
                    "purchase_date": purchase_date,
                    "purchase_fx_to_display": fx,
                    "original_tx_id": tx_id,
                }
            )
        open_lots[holding_keys[h]] = lots

    realized_gains_records: List[Dict[str, Any]] = []
    sell_events = (i for i, k in enumerate(kinds) if k == _FIFO_SELL)
    for i, (
        proceeds_local,
        cost_local,
        realized_gain_local,
        proceeds_display,
        cost_display,
        realized_gain_display,
    ) in zip(sell_events, sales):
        record = context[i]
        local_curr = record["LocalCurrency"]
        # Fallback logic for display gain
        if pd.isna(realized_gain_display) and pd.notna(realized_gain_local):
            if current_fx_rates_vs_usd:
                try:
                    fallback_rate = get_conversion_rate(
                        local_curr, display_currency, current_fx_rates_vs_usd
                    )
                    if pd.notna(fallback_rate):
                        realized_gain_display = realized_gain_local * fallback_rate
                except Exception:
                    pass
            if pd.isna(realized_gain_display) and local_curr == display_currency:
                realized_gain_display = realized_gain_local
        record["Total Proceeds (Local)"] = proceeds_local
        record["Total Cost Basis (Local)"] = cost_local
        record["Realized Gain (Local)"] = realized_gain_local
        record["Total Proceeds (Display)"] = proceeds_display
        record["Total Cost Basis (Display)"] = cost_display
        record["Realized Gain (Display)"] = realized_gain_display
        realized_gains_records.append(record)

    if not realized_gains_records:
        return pd.DataFrame(columns=output_columns), open_lots

    df_gains = pd.DataFrame(realized_gains_records, columns=output_columns)
    df_gains.sort_values(by=["Date", "Symbol", "Account"], inplace=True)

    return df_gains, open_lots


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
