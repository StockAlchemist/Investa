# Frozen reference: get_cash_flows_for_symbol_account and get_cash_flows_for_mwr
# from src/finutils.py at commit fed1eaa9, before the Rust port.
# tests/test_rust_cashflow_parity.py holds rust/investa_core/src/cashflows.rs
# (and the ledger around it) to them, bitwise and type for type. Never edit
# this to follow a change in Rust.
# ruff: noqa
import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CASH_SYMBOL_CSV, SHORTABLE_SYMBOLS
from finutils import get_conversion_rate, is_external_flow_row


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

    Filters transactions for the given symbol and account. Calculates the cash flow
    amount in the holding's local currency for each relevant transaction (buy, sell,
    dividend, fees, shorting actions). Appends the final market value (in local currency)
    as the last cash flow on the end_date.

    Args:
        symbol (str): The stock or ETF symbol.
        account (str): The investment account name.
        transactions (pd.DataFrame): The transactions DataFrame, filtered for the relevant
                                     period and scope. Must contain 'Date', 'Symbol', 'Account',
                                     'Type', 'Quantity', 'Price/Share', 'Commission',
                                     'Total Amount', and 'Local Currency' columns.
        final_market_value_local (float): The final market value of the holding in its
                                          local currency as of the end_date.
        end_date (date): The end date for the calculation period.

    Returns:
        Tuple[List[date], List[float]]: A tuple containing:
            - List[date]: A list of dates for the cash flows, sorted chronologically.
            - List[float]: A list of cash flows in the local currency. Buys/fees are negative,
                           sells/dividends are positive. The final market value is added
                           as a positive flow on the end_date.
            Returns ([], []) if no relevant transactions or final value exist, or if the
            cash flow pattern is invalid for IRR.
    """
    # --- MODIFIED: Handle missing 'To Account' column gracefully ---
    if account is None:
        symbol_account_tx_filtered = transactions_df[
            (transactions_df["Symbol"] == symbol)
        ]
    elif "To Account" in transactions_df.columns:
        account_mask = (transactions_df["Account"] == account) | (
            transactions_df["To Account"] == account
        )
        # --- ADDED: Check Note for account name (handles legacy transfers where Account col is misaligned) ---
        if "Note" in transactions_df.columns:
            account_mask |= transactions_df["Note"].str.contains(
                account, case=False, na=False
            ) & (transactions_df["Type"].str.lower() == "transfer")

        symbol_account_tx_filtered = transactions_df[
            (transactions_df["Symbol"] == symbol) & account_mask
        ]
    else:
        # If 'To Account' is missing, fallback to filtering only by 'Account'
        account_mask = transactions_df["Account"] == account
        if "Note" in transactions_df.columns:
            account_mask |= transactions_df["Note"].str.contains(
                account, case=False, na=False
            ) & (transactions_df["Type"].str.lower() == "transfer")

        symbol_account_tx_filtered = transactions_df[
            (transactions_df["Symbol"] == symbol) & account_mask
        ]
    # --- END MODIFIED ---
    if symbol_account_tx_filtered.empty:
        return [], []
    symbol_account_tx = symbol_account_tx_filtered.copy()
    dates_flows = defaultdict(float)
    symbol_account_tx.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )

    last_seen_price = None
    for _, row in symbol_account_tx.iterrows():
        tx_type = str(row.get("Type", "")).lower().strip()
        qty_val = row.get("Quantity")
        price_val = row.get("Price/Share")
        commission_val = row.get("Commission")
        total_amount_val = row.get("Total Amount")
        qty = pd.to_numeric(qty_val, errors="coerce")
        price_local = pd.to_numeric(price_val, errors="coerce")
        commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
        commission_local = (
            0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        )
        total_amount_local = pd.to_numeric(total_amount_val, errors="coerce")

        if pd.notna(price_local) and price_local > 1e-9:
            last_seen_price = price_local
        elif pd.notna(total_amount_local) and pd.notna(qty) and abs(qty) > 1e-9:
            last_seen_price = abs(total_amount_local / qty)
        tx_date = row["Date"].date()
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        total_amount_abs = (
            abs(total_amount_local) if pd.notna(total_amount_local) else 0.0
        )
        qty_val_abs = abs(qty) if pd.notna(qty) else 0.0
        cash_amt = total_amount_abs if total_amount_abs > 1e-9 else qty_val_abs

        if symbol == CASH_SYMBOL_CSV:
            if tx_type == "buy" or tx_type == "deposit":
                cash_flow_local = -(cash_amt + commission_local)
            elif tx_type == "sell" or tx_type == "withdrawal":
                cash_flow_local = cash_amt - commission_local
        else:
            if tx_type == "buy" or tx_type == "deposit":
                if pd.notna(qty) and qty > 0 and pd.notna(price_local):
                    cash_flow_local = -((qty_abs * price_local) + commission_local)
            elif tx_type == "sell" or tx_type == "withdrawal":
                if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                    cash_flow_local = (qty_abs * price_local) - commission_local

        if tx_type == "transfer" and is_transfer_a_flow:
            # If transfers are treated as flows, an incoming transfer is like a "buy" (outflow of cash to acquire asset)
            # and an outgoing transfer is like a "sell" (inflow of cash as asset is disposed).
            # We determine if it's incoming or outgoing based on the 'To Account' column.
            to_account = str(row.get("To Account", "")).strip()

            if symbol == CASH_SYMBOL_CSV:
                flow = cash_amt
            else:
                # Fallback for missing price in transfers
                price_to_use = price_local
                if (
                    pd.isna(price_to_use) or price_to_use <= 1e-9
                ) and last_seen_price is not None:
                    price_to_use = last_seen_price

                flow = qty * price_to_use if pd.notna(price_to_use) else 0.0

            # Determine direction if Account matches but To Account doesn't, or vice-versa, or Note matches
            acct_col = str(row.get("Account", "")).strip()
            note = str(row.get("Note", "")).lower()

            # Refined direction detection
            is_from_note = f"from {account.lower()}" in note
            is_to_note = f"to {account.lower()}" in note

            is_incoming = (to_account == account) or is_to_note
            is_outgoing = (acct_col == account) or is_from_note

            if is_incoming and not is_outgoing:
                # This is a transfer IN to the specified account. Treat as a cash outflow (like a buy).
                cash_flow_local = -flow
            elif is_outgoing and not is_incoming:
                # This is a transfer OUT of the specified account. Treat as a cash inflow (like a sell).
                cash_flow_local = flow
            else:
                # Ambiguous or both columns match (internal). Fallback to To Account check.
                if to_account == account:
                    cash_flow_local = -flow
                else:
                    cash_flow_local = flow
        elif tx_type == "short sell" and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = (qty_abs * price_local) - commission_local
        elif tx_type == "buy to cover" and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == "dividend":
            dividend_amount_local_cf = 0.0
            if pd.notna(total_amount_local):
                dividend_amount_local_cf = total_amount_local
            elif pd.notna(price_local):
                dividend_amount_local_cf = (
                    (qty_abs * price_local)
                    if (pd.notna(qty) and qty_abs > 0)
                    else price_local
                )
            if pd.notna(dividend_amount_local_cf):
                cash_flow_local = dividend_amount_local_cf - commission_local
        elif tx_type == "fees":
            cash_flow_local = -(abs(commission_local))
        elif tx_type in ["split", "stock split"]:
            cash_flow_local = 0.0
            if commission_local != 0:
                cash_flow_local = -abs(commission_local)

        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            try:
                flow_to_add = float(cash_flow_local)
                dates_flows[tx_date] += flow_to_add
            except (ValueError, TypeError):
                logging.warning(
                    f"Warning IRR CF Gen ({symbol}/{account}): Could not convert cash_flow_local {cash_flow_local} to float. Skipping flow."
                )

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
            first_tx_date_for_holding = (
                symbol_account_tx["Date"].min().date()
                if not symbol_account_tx.empty
                else report_date
            )
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

    Processes transactions for a single account, calculating the cash flow impact of each
    transaction (buys, sells, dividends, fees, cash deposits/withdrawals) in its local currency.
    Converts these local currency flows to the `target_currency`.

    CRITICAL CHANGE: Uses `historical_fx_rates` (if available) to convert flows using a pre-calculated
    rate specific to the transaction date. The `historical_fx_rates` dict is expected to map
    (date, local_currency) -> conversion_rate_to_target.
    This ensures accurate IRR for non-USD currencies by capturing FX fluctuations.
    Falls back to `fx_rates` (current/static) if historical data is missing.

    Args:
        account_transactions (pd.DataFrame): The transactions DataFrame filtered for a specific account.
        final_account_market_value (float): The final market value of the entire account in the
                                            `target_currency` as of the end_date.
        end_date (date): The end date for the calculation period.
        target_currency (str): The target currency for the MWR calculation.
        fx_rates (Optional[Dict[str, float]]): Current FX rates (fallback).
        display_currency (str): The display currency (informational).
        historical_fx_rates (Optional[Dict[Tuple[date, str], float]]): Dictionary mapping (Date, LocalCurrency)
            to the conversion rate to the target currency.

    Returns:
        Tuple[List[date], List[float]]: Sorted dates and cash flows in target currency.
    """
    if account_transactions.empty:
        return [], []
    acc_tx_copy = account_transactions.copy()
    dates_flows = defaultdict(float)
    acc_tx_copy.sort_values(by=["Date", "original_index"], inplace=True, ascending=True)

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

    for _, row in acc_tx_copy.iterrows():
        tx_type = str(row["Type"]).lower().strip()
        symbol = row["Symbol"]
        qty = pd.to_numeric(row["Quantity"], errors="coerce")
        price_local = pd.to_numeric(row["Price/Share"], errors="coerce")
        commission_val = row.get("Commission")
        total_amount_local = pd.to_numeric(row.get("Total Amount"), errors="coerce")
        commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
        commission_local = (
            0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        )
        tx_date = row["Date"].date()

        # --- PHASE 2 BUG FIX: Skip any transactions that occur after the evaluated end_date ---
        if tx_date > end_date:
            continue

        local_currency = row["Local Currency"]
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        # Under "portfolio" flow basis, buys/sells/dividends/short-sells/buy-to-cover/fees
        # are INTERNAL rotations of cash within the portfolio and emit NO external flow.
        # Only $CASH deposit/withdrawal and scope-crossing transfers contribute flows.
        # Under "per_trade" basis (default, used by symbol-level IRR), every trade is a
        # flow from that symbol/account's perspective.
        portfolio_basis = flow_basis == "portfolio"

        if symbol != CASH_SYMBOL_CSV:
            if portfolio_basis:
                if tx_type == "transfer":
                    is_outbound = False
                    is_inbound = False
                    acct = str(row.get("Account", "")).strip().upper()
                    to_acct = str(row.get("To Account", "")).strip().upper()

                    if acct in included_set:
                        is_outbound = True
                    if to_acct and to_acct in included_set:
                        is_inbound = True

                    if is_outbound and not is_inbound:
                        if pd.notna(qty) and pd.notna(price_local):
                            cash_flow_local = (abs(qty) * price_local) - abs(
                                commission_local
                            )
                    elif not is_outbound and is_inbound:
                        if pd.notna(qty) and pd.notna(price_local):
                            cash_flow_local = -(abs(qty) * price_local)
                # buy/sell/dividend/short sell/buy to cover/fees/tax/interest: 0.0 (internal)
            elif tx_type == "buy" or tx_type == "deposit":
                # External asset contribution or purchase
                if pd.notna(qty) and pd.notna(price_local):
                    cash_flow_local = -(
                        (qty_abs * price_local) + commission_local
                    )  # OUT from pocket (-)
            elif tx_type == "sell" or tx_type == "withdrawal":
                # External asset removal or sale
                if pd.notna(qty) and pd.notna(price_local):
                    cash_flow_local = (
                        qty_abs * price_local
                    ) - commission_local  # IN to pocket (+)
            elif tx_type == "dividend":
                dividend_amount_local_cf = 0.0
                if pd.notna(total_amount_local):
                    dividend_amount_local_cf = total_amount_local
                elif pd.notna(price_local):
                    dividend_amount_local_cf = (
                        (qty_abs * price_local)
                        if (pd.notna(qty) and qty_abs > 0)
                        else price_local
                    )
                if pd.notna(dividend_amount_local_cf):
                    cash_flow_local = dividend_amount_local_cf - commission_local
            elif tx_type == "short sell":
                if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                    cash_flow_local = (qty_abs * price_local) - commission_local
            elif tx_type == "buy to cover":
                if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                    cash_flow_local = -((qty_abs * price_local) + commission_local)
            elif tx_type == "fees":
                cash_flow_local = -abs(
                    commission_local
                )  # BUG-08 FIX: Use abs() consistently
            elif tx_type == "transfer":
                # Asset Transfer logic
                is_outbound = False
                is_inbound = False
                acct = str(row.get("Account", "")).strip().upper()
                to_acct = str(row.get("To Account", "")).strip().upper()

                if acct in included_set:
                    is_outbound = True
                if to_acct and to_acct in included_set:
                    is_inbound = True

                if is_outbound and not is_inbound:
                    # Asset leaving scope -> Withdrawal -> Positive MWR Flow
                    if pd.notna(qty) and pd.notna(price_local):
                        cash_flow_local = (abs(qty) * price_local) - abs(
                            commission_local
                        )  # BUG-09 FIX: Subtract commission (broker takes it)
                elif not is_outbound and is_inbound:
                    # Asset entering scope -> Deposit -> Negative MWR Flow
                    if pd.notna(qty) and pd.notna(price_local):
                        cash_flow_local = -(abs(qty) * price_local)

        elif symbol == CASH_SYMBOL_CSV:
            total_amount_abs = (
                abs(total_amount_local) if pd.notna(total_amount_local) else 0.0
            )
            qty_val_abs = abs(qty) if pd.notna(qty) else 0.0
            cash_amt = total_amount_abs if total_amount_abs > 1e-9 else qty_val_abs

            if portfolio_basis:
                # Portfolio-level MWR: use the centralized external-flow
                # classifier (see is_external_flow_row above). Convention is
                # "always external", so Note content is NOT inspected:
                #   - $CASH Deposit/Withdrawal → flow, whatever the Note
                #     (including "Auto-generated:" synthetic per-trade entries)
                #   - $CASH buy/sell (settlement-style bookkeeping) → 0
                #   - $CASH dividend/interest → 0 (internal income)
                # Filtering the synthetic rows out inflates early-year returns;
                # see tests/test_external_flow_classifier.py for the rationale.
                # Same definition as the TWR engine — see
                # _calculate_daily_net_cash_flow_vectorized in portfolio_cashflows.py.
                if is_external_flow_row(symbol, tx_type, row.get("Note", "")):
                    if tx_type == "deposit":
                        cash_flow_local = -(cash_amt + commission_local)
                    elif tx_type == "withdrawal":
                        cash_flow_local = cash_amt - commission_local
                # else: 0 (internal trade rotation, synthetic, or income on cash)
            elif tx_type in ["deposit", "buy"]:
                # External cash deposit (legacy per_trade)
                cash_flow_local = -(cash_amt + commission_local)  # OUT from pocket (-)
            elif tx_type in ["withdrawal", "sell"]:
                # External cash withdrawal (legacy per_trade)
                cash_flow_local = cash_amt - commission_local  # IN to pocket (+)
            elif tx_type in ["dividend", "interest"]:
                # Internal interest/dividends on cash are NOT external flows
                cash_flow_local = 0.0
            elif tx_type == "transfer":
                # Cash Transfer logic
                is_outbound = False
                is_inbound = False
                acct = str(row.get("Account", "")).strip().upper()
                to_acct = str(row.get("To Account", "")).strip().upper()

                if acct in included_set:
                    is_outbound = True
                if to_acct and to_acct in included_set:
                    is_inbound = True

                if is_outbound and not is_inbound:
                    # Money leaving the scope -> Withdrawal -> Positive MWR Flow
                    cash_flow_local = cash_amt - commission_local
                elif not is_outbound and is_inbound:
                    # Money entering the scope -> Deposit -> Negative MWR Flow
                    cash_flow_local = -(cash_amt + commission_local)

        cash_flow_target = cash_flow_local
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            if local_currency != target_currency:
                rate = np.nan
                # Try Historical Lookup
                if historical_fx_rates:
                    lookup_date = tx_date
                    if isinstance(lookup_date, pd.Timestamp):
                        lookup_date = lookup_date.date()
                    elif isinstance(lookup_date, datetime):
                        lookup_date = lookup_date.date()
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
