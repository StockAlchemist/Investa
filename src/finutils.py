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
from datetime import date, datetime, timedelta
import hashlib  # noqa: F401 -- read by the Rust port at call time
import weakref
import config
import logging
import investa_core
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict

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


class CashFlowLedger:
    """A transactions frame read once, for any number of per-holding IRR flows.

    `get_cash_flows_for_symbol_account` re-filtered the whole frame on every
    call: an equality mask, a case-insensitive regex over every Note, and an
    iterrows pass. That cost a summary about half its time (243 holdings,
    780,000 regex searches). Here the rows are grouped by symbol once, each
    account's mask is built once with the original's own pandas expressions,
    and each row is parsed once. The flow rules run in investa_core.

    Nothing is read until the first request, so a frame the original could
    not read fails inside the caller's handler exactly as before.
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df
        self._by_symbol: Optional[Dict[str, List[int]]] = None
        self._account_masks: Dict[Any, np.ndarray] = {}
        self._parsed = None
        self._is_transfer: Optional[Tuple[bool, Any]] = None
        self._ids: Optional[list] = None

    def _symbol_positions(self, symbol: Any) -> List[int]:
        if not isinstance(symbol, str):
            return np.flatnonzero((self.df["Symbol"] == symbol).to_numpy()).tolist()
        if self._by_symbol is None:
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, s in enumerate(self.df["Symbol"].tolist()):
                if isinstance(s, str):
                    groups[s].append(i)
            self._by_symbol = groups
        return self._by_symbol.get(symbol, [])

    def _account_mask(self, account: Any) -> np.ndarray:
        if account in self._account_masks:
            return self._account_masks[account]
        df = self.df
        if "To Account" in df.columns:
            mask = (df["Account"] == account) | (df["To Account"] == account)
        else:
            mask = df["Account"] == account
        # Legacy transfers whose Account column is misaligned name the account
        # in the Note. str.contains reads the account name as a regex.
        if "Note" in df.columns:
            mask |= (
                df["Note"].str.contains(account, case=False, na=False)
                & self._transfer_rows()
            )
        mask = mask.to_numpy(dtype=bool)
        self._account_masks[account] = mask
        return mask

    def _transfer_rows(self) -> pd.Series:
        """`df["Type"].str.lower() == "transfer"`, the same for every account."""
        if self._is_transfer is None:
            try:
                self._is_transfer = (True, self.df["Type"].str.lower() == "transfer")
            except Exception as exc:  # raised again for every account, as before
                self._is_transfer = (False, exc)
        ok, value = self._is_transfer
        if not ok:
            raise value
        return value

    def _rows(self):
        if self._parsed is None:
            df = self.df
            columns = df.columns

            def texts(name):
                return df[name].tolist() if name in columns else None

            types = texts("Type")
            self._parsed = {
                "kind": np.array(
                    [
                        _FLOW_KINDS.get(_cell_text(types, i).lower().strip(), 0)
                        for i in range(len(df))
                    ],
                    dtype=np.uint8,
                ),
                "qty": _numeric_column(df, "Quantity"),
                "price": _numeric_column(df, "Price/Share"),
                "commission": _commission_column(df),
                "total": _numeric_column(df, "Total Amount"),
                "date": df["Date"].tolist(),
                "account": texts("Account"),
                "to_account": texts("To Account"),
                "note": texts("Note"),
            }
        return self._parsed

    def _sorted(self, positions: List[int]) -> List[int]:
        """The positions in the original's `sort_values(["Date", "original_index"])`
        order: a stable sort, missing keys last."""
        if "original_index" not in self.df.columns:
            raise KeyError("original_index")
        dates = self._rows()["date"]
        if self._ids is None:
            self._ids = self.df["original_index"].tolist()
        id_values = self._ids
        clean = all(
            isinstance(dates[p], pd.Timestamp) and dates[p] is not pd.NaT
            for p in positions
        ) and all(isinstance(id_values[p], (int, np.integer)) for p in positions)
        if clean:
            return sorted(positions, key=lambda p: (dates[p], id_values[p]))
        keys = self.df.iloc[positions][["Date", "original_index"]].copy()
        keys["_position"] = positions
        return keys.sort_values(by=["Date", "original_index"])["_position"].tolist()

    def flows_for(
        self,
        symbol: str,
        account: Optional[str],
        final_market_value: float,
        is_transfer_a_flow: bool,
        report_date: Optional[date] = None,
    ) -> Tuple[List[date], List[float]]:
        """`get_cash_flows_for_symbol_account` for one holding."""
        positions = self._symbol_positions(symbol)
        if account is not None:
            mask = self._account_mask(account)
            positions = [p for p in positions if mask[p]]
        if not positions:
            return [], []
        order = self._sorted(positions)
        rows = self._rows()
        take = np.asarray(order, dtype=np.intp)
        kind = rows["kind"][take]
        tx_dates = [rows["date"][p].date() for p in order]

        # A transfer's direction, when it counts as a flow: into the account
        # (by To Account or "to <account>" in the Note) is money in (-), out
        # of it is money out (+); when both or neither match, To Account decides.
        direction = np.zeros(len(order), dtype=np.int8)
        if is_transfer_a_flow:
            for j, p in enumerate(order):
                if kind[j] != _TRANSFER:
                    continue
                to_account = _cell_text(rows["to_account"], p).strip()
                acct_col = _cell_text(rows["account"], p).strip()
                note = _cell_text(rows["note"], p).lower()
                is_from_note = f"from {account.lower()}" in note
                is_to_note = f"to {account.lower()}" in note
                is_incoming = (to_account == account) or is_to_note
                is_outgoing = (acct_col == account) or is_from_note
                if is_incoming and not is_outgoing:
                    direction[j] = -1
                elif is_outgoing and not is_incoming:
                    direction[j] = 1
                else:
                    direction[j] = -1 if to_account == account else 1

        flows = investa_core.holding_flows(
            kind.tolist(),
            rows["qty"][take].tolist(),
            rows["price"][take].tolist(),
            rows["commission"][take].tolist(),
            rows["total"][take].tolist(),
            direction.tolist(),
            bool(symbol == CASH_SYMBOL_CSV),
            symbol in SHORTABLE_SYMBOLS,
            bool(is_transfer_a_flow),
        )
        dates_flows = defaultdict(float)
        for tx_date, cash_flow_local in zip(tx_dates, flows):
            if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
                dates_flows[tx_date] += float(cash_flow_local)

        sorted_dates = sorted(dates_flows.keys())
        if not sorted_dates and abs(final_market_value) < 1e-9:
            return [], []
        final_dates = list(sorted_dates)
        final_flows = [float(dates_flows[d]) for d in final_dates]
        final_market_value_abs = (
            abs(final_market_value) if pd.notna(final_market_value) else 0.0
        )
        if final_market_value_abs > 1e-9 and isinstance(report_date, date):
            if not final_dates:
                first_tx_date_for_holding = self.df["Date"].iloc[order].min().date()
                if report_date >= first_tx_date_for_holding:
                    final_dates.append(report_date)
                    final_flows.append(final_market_value_abs)
                else:
                    return [], []
            elif report_date >= final_dates[-1]:
                if final_dates[-1] == report_date:
                    final_flows[-1] += final_market_value_abs
                else:
                    final_dates.append(report_date)
                    final_flows.append(final_market_value_abs)

        if len(final_dates) < 2:
            return [], []
        non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
        if (
            not non_zero_final_flows
            or all(cf >= -1e-9 for cf in non_zero_final_flows)
            or all(cf <= 1e-9 for cf in non_zero_final_flows)
        ):
            return [], []
        return final_dates, final_flows


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
    return CashFlowLedger(transactions_df).flows_for(
        symbol, account, final_market_value, is_transfer_a_flow, report_date
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
    if account_transactions.empty:
        return [], []
    acc_tx = account_transactions.sort_values(
        by=["Date", "original_index"], ascending=True
    )

    # Pre-calculate the scope of accounts for "Internal vs External" flow detection.
    # Transfers between accounts WITHIN this scope are 0-flow (internal).
    # Transfers to/from accounts OUTSIDE this scope are external flows (deposits/withdrawals).
    if include_accounts:
        included_set = {str(a).strip().upper() for a in include_accounts if a}
    else:
        # If no explicit list, the scope is the entire set of accounts in the provided transactions.
        # This is critical for the "All Accounts" dashboard view.
        all_accs = set(account_transactions["Account"].dropna().unique())
        if "To Account" in account_transactions.columns:
            all_accs.update(account_transactions["To Account"].dropna().unique())
        included_set = {str(a).strip().upper() for a in all_accs if a}

    types = [str(t).lower().strip() for t in acc_tx["Type"].tolist()]
    symbols = acc_tx["Symbol"].tolist()
    acc_tx["Quantity"], acc_tx["Price/Share"]  # read strictly, as the original did
    qty, qty_np = _numeric_column(acc_tx, "Quantity", with_types=True)
    price, price_np = _numeric_column(acc_tx, "Price/Share", with_types=True)
    commission = _commission_column(acc_tx)
    total, total_np = _numeric_column(acc_tx, "Total Amount", with_types=True)
    tx_dates = [d.date() for d in acc_tx["Date"].tolist()]

    # --- PHASE 2 BUG FIX: Skip any transactions that occur after the evaluated end_date ---
    kept = [i for i, d in enumerate(tx_dates) if not d > end_date]
    if not kept:
        local_currencies: list = []
    else:
        local_currencies = acc_tx["Local Currency"].tolist()

    accounts = acc_tx["Account"].tolist() if "Account" in acc_tx.columns else None
    to_accounts = (
        acc_tx["To Account"].tolist() if "To Account" in acc_tx.columns else None
    )
    kind = np.array([_FLOW_KINDS.get(types[i], 0) for i in kept], dtype=np.uint8)
    # A transfer crossing the scope: leaving it is money out (+), entering it
    # money in (-); within it, or outside it entirely, no flow.
    direction = np.zeros(len(kept), dtype=np.int8)
    for j, i in enumerate(kept):
        if kind[j] != _TRANSFER:
            continue
        acct = _cell_text(accounts, i).strip().upper()
        to_acct = _cell_text(to_accounts, i).strip().upper()
        is_outbound = acct in included_set
        is_inbound = bool(to_acct) and to_acct in included_set
        if is_outbound and not is_inbound:
            direction[j] = 1
        elif not is_outbound and is_inbound:
            direction[j] = -1

    take = np.asarray(kept, dtype=np.intp)
    flows = investa_core.mwr_flows(
        kind.tolist(),
        qty[take].tolist(),
        price[take].tolist(),
        commission[take].tolist(),
        total[take].tolist(),
        direction.tolist(),
        [bool(symbols[i] == CASH_SYMBOL_CSV) for i in kept],
        flow_basis == "portfolio",
    )

    dates_flows = defaultdict(float)
    for i, (cash_flow_local, uses) in zip(kept, flows):
        if uses and (
            (uses & 1 and qty_np[i])
            or (uses & 2 and price_np[i])
            or (uses & 4 and total_np[i])
        ):
            cash_flow_local = np.float64(cash_flow_local)
        tx_date = tx_dates[i]
        local_currency = local_currencies[i]
        cash_flow_target = cash_flow_local
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            if local_currency != target_currency:
                rate = np.nan
                # Try Historical Lookup
                if historical_fx_rates:
                    lookup_date = tx_date
                    if (lookup_date, local_currency) in historical_fx_rates:
                        rate = historical_fx_rates[(lookup_date, local_currency)]

                # Fallback to Current Rate
                if pd.isna(rate):
                    rate = get_conversion_rate(
                        local_currency, target_currency, fx_rates
                    )

                if pd.isna(rate):
                    logging.warning(
                        f"MWR calc conversion failed for {local_currency} on {tx_date}. Skipping."
                    )
                    cash_flow_target = 0.0
                else:
                    cash_flow_target = cash_flow_local * rate

            if pd.notna(cash_flow_target) and abs(cash_flow_target) > 1e-9:
                dates_flows[tx_date] += cash_flow_target

    sorted_dates = sorted(dates_flows.keys())
    if not sorted_dates and abs(final_account_market_value) < 1e-9:
        return [], []

    final_dates = list(sorted_dates)
    final_flows = [dates_flows[d] for d in final_dates]
    final_market_value_target = (
        float(final_account_market_value)
        if pd.notna(final_account_market_value)
        else 0.0
    )
    final_market_value_abs = abs(final_market_value_target)

    if final_market_value_abs > 1e-9 and isinstance(end_date, date):
        if not final_dates:
            return [], []
        elif end_date >= final_dates[-1]:
            if final_dates[-1] == end_date:
                final_flows[-1] += final_market_value_abs
            else:
                final_dates.append(end_date)
                final_flows.append(final_market_value_abs)

    if len(final_dates) < 2:
        return [], []

    non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
    if (
        not non_zero_final_flows
        or all(cf >= -1e-9 for cf in non_zero_final_flows)
        or all(cf <= 1e-9 for cf in non_zero_final_flows)
    ):
        return [], []
    return final_dates, final_flows


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
    if (
        not from_curr
        or not isinstance(from_curr, str)
        or not to_curr
        or not isinstance(to_curr, str)
    ):
        logging.debug("get_conversion_rate: Invalid from_curr or to_curr input.")
        return np.nan  # Return NaN on invalid input
    if from_curr == to_curr:
        return 1.0
    if not isinstance(fx_rates, dict):
        # Changed to debug as this might be expected in some initial states
        # or if no FX data was fetched at all.
        logging.debug(
            "Warning: get_conversion_rate received invalid fx_rates type. Returning NaN"
        )
        return np.nan  # Return NaN on invalid rates dict

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()
    rate_A_per_USD = fx_rates.get(from_curr_upper)
    if rate_A_per_USD is None:
        pair_A = "USDTHB=X" if from_curr_upper == "THB" else f"{from_curr_upper}=X"
        rate_A_per_USD = fx_rates.get(pair_A)

    # --- NEW: Static Fallback Injection ---
    # If a rate is missing from the provided API-based dictionary, try the static fallback.
    if pd.isna(rate_A_per_USD) and hasattr(config, "STATIC_FX_FALLBACK"):
        rate_A_per_USD = config.STATIC_FX_FALLBACK.get(from_curr_upper)
        if pd.notna(rate_A_per_USD):
            logging.info(f"FX Fallback: Using STATIC rate for {from_curr_upper}")

    # Added debug log for lookup
    logging.debug(
        f"get_conversion_rate: Looking up {from_curr_upper}/USD (key '{from_curr_upper}'): {rate_A_per_USD}"
    )
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0
    rate_B_per_USD = fx_rates.get(to_curr_upper)
    if rate_B_per_USD is None:
        pair_B = "USDTHB=X" if to_curr_upper == "THB" else f"{to_curr_upper}=X"
        rate_B_per_USD = fx_rates.get(pair_B)

    if pd.isna(rate_B_per_USD) and hasattr(config, "STATIC_FX_FALLBACK"):
        rate_B_per_USD = config.STATIC_FX_FALLBACK.get(to_curr_upper)
        if pd.notna(rate_B_per_USD):
            logging.info(f"FX Fallback: Using STATIC rate for {to_curr_upper}")
    # --------------------------------------

    # Added debug log for lookup
    logging.debug(
        f"get_conversion_rate: Looking up {to_curr_upper}/USD (key '{to_curr_upper}'): {rate_B_per_USD}"
    )
    if to_curr_upper == "USD":
        rate_B_per_USD = 1.0

    rate_B_per_A = np.nan
    if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
        if abs(rate_A_per_USD) > 1e-9:
            try:
                rate_B_per_A = rate_B_per_USD / rate_A_per_USD
                logging.debug(  # Keep debug log for successful calculation
                    f"DEBUG get_conv_rate: {to_curr}/{from_curr} = {rate_B_per_USD} / {rate_A_per_USD} = {rate_B_per_A}"
                )
            except (ZeroDivisionError, TypeError):
                pass

    if pd.isna(rate_B_per_A):
        # Log the warning but return a neutral rate (1.0) to avoid breaking calculations?
        # Or return NaN to signal failure? Let's return NaN for clarity.
        logging.warning(  # Keep warning for failed conversion
            f"Warning: Current FX rate lookup failed for {from_curr}->{to_curr}. Returning NaN"
        )
        return np.nan  # Fallback to NaN to indicate failure
        # return 1.0  # Fallback to 1.0 might hide errors
    else:
        return float(rate_B_per_A)


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
    key = id(df)
    signature = (len(df), tuple(df.columns), df.index[0], df.index[-1])
    hit = _PRICE_TABLES.get(key)
    if hit is not None and hit[0]() is df and hit[1] == signature:
        return hit[2]
    table = _build_price_table(df)
    try:
        ref = weakref.ref(df, lambda _r, k=key: _PRICE_TABLES.pop(k, None))
    except TypeError:
        return table
    _PRICE_TABLES[key] = (ref, signature, table)
    return table


def _build_price_table(df: pd.DataFrame) -> Optional["investa_core.PriceTable"]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    if idx.tz is not None and str(idx.tz) != "UTC":
        return None  # midnight in another zone: leave the localisation to pandas
    if not df.columns.is_unique:
        return None
    columns = [c for c in ("price", "Adj Close", "Close") if c in df.columns]
    if not columns:
        return None
    column = df[columns[0]]
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        return None
    if not idx.is_monotonic_increasing:
        df = df.sort_index()  # as the original sorts before its asof
        column = df[columns[0]]
    valid = ~df.isna().any(axis=1).to_numpy()
    try:
        ticks = df.index.as_unit("ns").asi8  # an index may count in s/ms/us
    except (ValueError, OverflowError):
        return None
    values = column.to_numpy(dtype=np.float64)
    return investa_core.PriceTable(
        np.ascontiguousarray(ticks[valid]),
        np.ascontiguousarray(values[valid]),
        int(ticks[0]),
        float(values[0]),
    )


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
    if target_date is None or not isinstance(
        target_date, (date, datetime, pd.Timestamp)
    ):
        return np.nan

    if symbol_key not in prices_dict or prices_dict[symbol_key].empty:
        # logging.debug(f"get_historical_price: {symbol_key} missing or empty.")
        return np.nan

    # Avoid modifying the original DataFrame in prices_dict (it might be shared)
    df = prices_dict[symbol_key]

    # A calendar day against a frame the Rust table covers: binary search
    # instead of a pandas asof call (same answer, see _price_table).
    if type(target_date) is date:
        ns = (target_date.toordinal() - _EPOCH_ORDINAL) * _DAY_NS
        if _TS_MIN_NS <= ns <= _TS_MAX_NS:
            table = _price_table(df)
            if table is not None:
                return table.asof(ns)

    try:
        # Optimization: Use asof for efficient lookup
        # If not sorted, sort into a NEW object (do not modify inplace)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Try asof with original target_date
        try:
            # Force target_date to Timestamp for more reliable awareness checks in pandas
            target_ts = pd.Timestamp(target_date)

            # Align awareness and type BEFORE calling asof
            if isinstance(df.index, pd.DatetimeIndex):
                idx_tz = df.index.tz
                target_tz = target_ts.tz

                if idx_tz is not None and target_tz is None:
                    target_ts = target_ts.tz_localize(idx_tz)
                elif idx_tz is None and target_tz is not None:
                    target_ts = target_ts.tz_localize(None)
                elif idx_tz is not None and target_tz is not None:
                    target_ts = target_ts.tz_convert(idx_tz)
            elif (
                len(df.index) > 0
                and isinstance(df.index[0], date)
                and not isinstance(df.index[0], datetime)
            ):
                # index is likely date objects, asof(Timestamp) might fail.
                # Convert target_ts back to date for lookup if index is date objects.
                target_ts = target_ts.date()

            res = df.asof(target_ts)

        except (TypeError, ValueError) as e_asof:
            logging.debug(
                f"AsOf initial fail for {symbol_key} on {target_date}: {e_asof}"
            )
            # Fallback: Try converting to Timestamp
            ts_target = pd.Timestamp(target_date)

            # Handle Timezone if index is tz-aware
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                if ts_target.tz is None:
                    ts_target = ts_target.tz_localize(df.index.tz)

            res = df.asof(ts_target)

        # Extract price from result
        price = np.nan
        if isinstance(res, pd.Series):
            price = res.get("price")
            if price is None or pd.isna(price):
                price = res.get("Adj Close")
            if price is None or pd.isna(price):
                price = res.get("Close")
        elif isinstance(res, pd.DataFrame):
            # Duplicate index case
            if not res.empty:
                if "price" in res.columns:
                    price = res["price"].iloc[-1]
                elif "Adj Close" in res.columns:
                    price = res["Adj Close"].iloc[-1]
                elif "Close" in res.columns:
                    price = res["Close"].iloc[-1]
        elif pd.notna(res):
            pass  # Scalar?

        if price is None or pd.isna(price):
            # Check if target_date is before the start of the data (Backfill Strategy)
            # Use Timestamps for robust comparison
            try:
                first_ts = pd.Timestamp(df.index[0])
                target_ts_eval = pd.Timestamp(target_date)

                # Align awareness for comparison
                if first_ts.tz is not None and target_ts_eval.tz is None:
                    target_ts_eval = target_ts_eval.tz_localize(first_ts.tz)
                elif first_ts.tz is None and target_ts_eval.tz is not None:
                    target_ts_eval = target_ts_eval.tz_localize(None)

                if target_ts_eval < first_ts:
                    col = "price"
                    if "price" not in df.columns:
                        if "Adj Close" in df.columns:
                            col = "Adj Close"
                        elif "Close" in df.columns:
                            col = "Close"

                    backfill_price = df[col].iloc[0]
                    if pd.notna(backfill_price):
                        # logging.debug(f"BACKFILL: {symbol_key} using {backfill_price} for {target_date}")
                        return float(backfill_price)
            except Exception as e_comp:
                logging.debug(f"Comparison fail in get_historical_price: {e_comp}")

        return float(price) if pd.notna(price) else np.nan

    except Exception as e:
        logging.error(
            f"ERROR getting historical price for {symbol_key} on {target_date}: {e}"
        )
        return np.nan


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
    if (
        not from_curr
        or not isinstance(from_curr, str)
        or not to_curr
        or not isinstance(to_curr, str)
        or not isinstance(target_date, date)
    ):
        logging.debug(  # Changed to debug as this might be expected for invalid inputs
            f"Hist FX Bridge: Invalid input - from={from_curr}, to={to_curr}, date={target_date}"
        )
        return np.nan
    if from_curr == to_curr:
        return 1.0
    if not isinstance(historical_fx_data, dict):
        logging.warning("Hist FX Bridge: Invalid historical_fx_data type received.")
        return np.nan  # Return NaN on invalid input dict

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()
    rate_A_per_USD = np.nan
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0
    else:
        pair_A = "USDTHB=X" if from_curr_upper == "THB" else f"{from_curr_upper}=X"
        logging.debug(
            f"Hist FX Bridge: Getting historical price for {pair_A} on {target_date}"
        )  # Added debug log
        price_A = get_historical_price(pair_A, target_date, historical_fx_data)
        if price_A is not None and pd.notna(price_A):
            rate_A_per_USD = price_A
    rate_B_per_USD = np.nan
    if to_curr_upper == "USD":
        rate_B_per_USD = 1.0
    else:
        pair_B = "USDTHB=X" if to_curr_upper == "THB" else f"{to_curr_upper}=X"
        logging.debug(
            f"Hist FX Bridge: Getting historical price for {pair_B} on {target_date}"
        )  # Added debug log
        price_B = get_historical_price(pair_B, target_date, historical_fx_data)
        if price_B is not None and pd.notna(price_B):
            rate_B_per_USD = price_B

    rate_B_per_A = np.nan
    if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
        if abs(rate_A_per_USD) > 1e-12:
            try:
                rate_B_per_A = rate_B_per_USD / rate_A_per_USD
            except (ZeroDivisionError, TypeError, OverflowError):
                pass

    if pd.isna(rate_B_per_A):
        if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
            logging.warning(  # Keep warning for calculation failure despite finding intermediate rates
                f"Hist FX Bridge: Calculation failed for {from_curr}->{to_curr} on {target_date} despite finding intermediate rates ({rate_A_per_USD}, {rate_B_per_USD})."
            )
        return np.nan
    else:
        return float(rate_B_per_A)


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

    if not internal_symbol or not isinstance(internal_symbol, str):
        return None

    # --- ADDED Logging ---
    logging.debug(f"map_to_yf_symbol: Received internal_symbol='{internal_symbol}'")

    # Normalize the input symbol (uppercase, strip whitespace)
    normalized_symbol = internal_symbol.upper().strip()
    logging.debug(f"  Normalized to: '{normalized_symbol}'")

    # --- 1. Check Excluded and Cash Symbols FIRST ---
    if is_cash_symbol(internal_symbol):
        logging.debug(f"  Symbol '{normalized_symbol}' is CASH. Returning None.")
        return None

    if normalized_symbol in user_excluded_symbols:
        logging.debug(
            f"  map_to_yf_symbol: Symbol '{normalized_symbol}' is in EXCLUSION list. Returning None."
        )
        return None

    # --- 2. Check Explicit Map (if not excluded) ---
    # Ensure keys in the map are also normalized for comparison
    normalized_map = {
        k.upper().strip(): v.upper().strip() for k, v in user_symbol_map.items()
    }  # Use user-defined map

    # ADDED: Default system mappings for common problematic tickers
    SYSTEM_MAP = {
        "BRK.B": "BRK-B",
        "BRK.A": "BRK-A",
        "BF.B": "BF-B",
        "BF.A": "BF-A",
        "RDS.A": "RDS-A",
        "RDS.B": "RDS-B",
    }

    if normalized_symbol in normalized_map:
        mapped_symbol = normalized_map[normalized_symbol]
        logging.debug(
            f"  Found in explicit map: '{normalized_symbol}' -> '{mapped_symbol}'"
        )
        return mapped_symbol

    if normalized_symbol in SYSTEM_MAP:
        mapped_symbol = SYSTEM_MAP[normalized_symbol]
        logging.debug(
            f"  Applied SYSTEM_MAP: '{normalized_symbol}' -> '{mapped_symbol}'"
        )
        return mapped_symbol

    # --- 3. Apply Automatic Conversion Rules (if not found in map) ---
    # Handle Thai stocks (:BKK -> .BK)
    if ":BKK" in normalized_symbol:
        base_symbol = normalized_symbol.replace(":BKK", "")
        # Handle cases like "ADVANC:BKK" -> "ADVANC.BK"
        if "." in base_symbol or len(base_symbol) == 0:
            logging.warning(
                f"map_to_yf_symbol: Skipping potentially invalid BKK conversion: {internal_symbol}"
            )
            return None
        return f"{base_symbol.upper()}.BK"

    # --- 4. Sanitization and Heuristics ---

    # Rule 4a: Replace dots with dashes for short US-style tickers (e.g. BRK.B -> BRK-B)
    # This catches variations not in the explicit SYSTEM_MAP.
    if "." in normalized_symbol and len(normalized_symbol) <= 6:
        # Check if it looks like a class (e.g. TKR.A)
        parts = normalized_symbol.split(".")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Explicitly skip known valid exchange suffixes like .BK
            if parts[1].upper() not in ["BK", "L", "TO", "V"]:
                converted = normalized_symbol.replace(".", "-")
                logging.debug(
                    f"  Applied DOT-to-DASH heuristic: '{normalized_symbol}' -> '{converted}'"
                )
                return converted

    # Rule 4b: Skip obviously invalid tickers (Custom names, fund tags, etc.)
    # Valid tickers typically only contain letters, numbers, and very few special chars (. - :)
    import re

    # Allowing : for BKK (handled above) and ^ for indices
    if not re.match(r"^[A-Z0-9\.\-\^:]+$", normalized_symbol):
        logging.warning(
            f"map_to_yf_symbol: Skipping symbol with invalid characters: {internal_symbol}"
        )
        return None

    # Rule 4c: Extremely long symbols (likely descriptive names or mutual fund tags)
    if len(normalized_symbol) > 15:
        logging.warning(
            f"map_to_yf_symbol: Skipping excessively long symbol: {internal_symbol}"
        )
        return None

    # --- 5. Return normalized symbol if no rules applied ---
    logging.debug(
        f"  No rules applied. Returning normalized symbol: '{normalized_symbol.upper()}'"
    )
    return normalized_symbol.upper()


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
    if col not in df.columns:
        return 0.0
    try:
        data_series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        total = data_series.sum()
        return float(total) if pd.notna(total) else 0.0
    except Exception as e:
        logging.error(f"Error in safe_sum for column {col}: {e}")
        return 0.0


# --- Value Formatting Utilities ---


def format_currency_value(
    value: Optional[float],
    currency_symbol: str = "$",
    decimals: int = 2,
    show_plus_sign: bool = False,
    is_abs: bool = False,  # If true, formats absolute value
) -> str:
    """Formats a numeric value as a currency string."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val_to_format = abs(float(value)) if is_abs else float(value)
        if (
            abs(val_to_format) < 1e-9 and not is_abs
        ):  # Treat near-zero as zero unless we want absolute display
            val_to_format = 0.0

        sign = ""
        if (
            show_plus_sign and val_to_format > 1e-9
        ):  # Only show plus for positive non-zero
            sign = "+"
        elif (
            val_to_format < -1e-9
        ):  # Negative sign is handled by f-string if not is_abs
            pass  # Negative sign will be part of the number

        formatted_num = f"{val_to_format:,.{decimals}f}"
        return (
            f"{sign}{currency_symbol}{formatted_num}"
            if not is_abs
            else f"{currency_symbol}{formatted_num}"
        )
    except (ValueError, TypeError):
        return "N/A"


def format_percentage_value(
    value: Optional[float], decimals: int = 2, show_plus_sign: bool = False
) -> str:
    """Formats a numeric value (decimal fraction) as a percentage string."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val_float = float(value)
        if np.isinf(val_float):
            return "Inf %"

        sign = ""
        # For percentages, value is already scaled (e.g., 25.0 for 25%).
        # If it's a factor (e.g., 0.25 for 25%), multiply by 100 first.
        # Assuming 'value' is already the percentage number (e.g., 2.5 for 2.5%)
        # If it's a factor (e.g. 0.025 for 2.5%), then it should be multiplied by 100.
        # Let's assume for now 'value' is the actual percentage number.
        # If it's consistently a factor, we'd add: val_float *= 100

        if show_plus_sign and val_float > 1e-9:
            sign = "+"

        return f"{sign}{val_float:,.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_large_number_display(
    value: Optional[float], currency_symbol: str = "$", decimals: int = 2
) -> str:
    """Formats large numbers with T, B, M suffixes (e.g., for Market Cap)."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        val_float = float(value)
        if val_float >= 1e12:
            return f"{currency_symbol}{val_float / 1e12:,.{decimals}f} T"
        elif val_float >= 1e9:
            return f"{currency_symbol}{val_float / 1e9:,.{decimals}f} B"
        elif val_float >= 1e6:
            return f"{currency_symbol}{val_float / 1e6:,.{decimals}f} M"
        else:
            return f"{currency_symbol}{val_float:,.{decimals}f}"  # Default to standard currency format
    except (ValueError, TypeError):
        return "N/A"


def format_integer_with_commas(value: Optional[float]) -> str:
    """Formats a number as an integer string with commas."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{int(round(float(value))):,}"
    except (ValueError, TypeError):
        return "N/A"


def format_float_with_commas(value: Optional[float], decimals: int = 2) -> str:
    """Formats a float string with commas and specified decimal places."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"
