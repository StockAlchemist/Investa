# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          finutils.py
 Purpose:       Financial and general utility functions for Investa Portfolio Dashboard.
                Includes IRR/NPV calculations, currency conversion helpers,
                cash flow generation, historical price lookups, and file hashing.

 Author:        Kit Matan (Derived from portfolio_logic.py) and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import hashlib
import os
import logging
from scipy import optimize
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict

# Import constants needed within these utility functions
# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    from config import (
        CASH_SYMBOL_CSV,
        SHORTABLE_SYMBOLS,
    )  # YFINANCE_EXCLUDED_SYMBOLS, SYMBOL_MAP_TO_YFINANCE removed
except ImportError:
    # Fallback values if config import fails (should not happen in normal execution)
    logging.error("CRITICAL: Could not import constants from config.py in finutils.py")
    CASH_SYMBOL_CSV = "$CASH"
    SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}


# --- Constants for _get_file_hash ---
HASH_CHUNK_SIZE = 8192
HASH_ERROR_NOT_FOUND = "FILE_NOT_FOUND"
HASH_ERROR_PERMISSION = "HASHING_ERROR_PERMISSION"
HASH_ERROR_IO = "HASHING_ERROR_IO"
HASH_ERROR_UNEXPECTED = "HASHING_ERROR_UNEXPECTED"


# --- File Hashing Helper ---
def _get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The SHA256 hash of the file as a hexadecimal string,
             or a specific error string ('FILE_NOT_FOUND',
             'HASHING_ERROR_PERMISSION', 'HASHING_ERROR_IO',
             'HASHING_ERROR_UNEXPECTED') if an error occurs.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(HASH_CHUNK_SIZE):  # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logging.warning(f"Warning: File not found for hashing: {filepath}")
        return HASH_ERROR_NOT_FOUND
    except PermissionError:
        logging.error(f"Permission denied accessing file {filepath} for hashing.")
        return HASH_ERROR_PERMISSION
    except IOError as e:
        logging.error(f"I/O error hashing file {filepath}: {e}")
        return HASH_ERROR_IO
    except Exception as e:
        logging.exception(f"Unexpected error hashing file {filepath}")
        return HASH_ERROR_UNEXPECTED


# --- IRR/MWR Calculation Functions ---
def calculate_npv(rate: float, dates: List[date], cash_flows: List[float]) -> float:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows.

    Discounts each cash flow back to the date of the first cash flow using the
    provided discount rate, assuming time is measured in years (days/365.0).
    Handles potential errors like invalid rates, date/flow mismatches, and
    calculation issues (e.g., division by zero, overflow).

    Args:
        rate (float): The discount rate per period (annualized).
        dates (List[date]): A list of dates corresponding to the cash flows. Must be sorted.
        cash_flows (List[float]): A list of cash flows. Must be the same length as dates.

    Returns:
        float: The calculated Net Present Value (NPV). Returns np.nan if inputs are invalid,
               lengths mismatch, dates are unsorted, or calculation errors occur.

    Raises:
        ValueError: If the lengths of `dates` and `cash_flows` do not match.
                    (This is primarily for the IRR solver using this function).
    """
    # Refined Error Handling
    if not isinstance(rate, (int, float)) or not np.isfinite(rate):
        # logging.debug("NPV Calculation Error: Invalid rate provided.")
        return np.nan
    if len(dates) != len(cash_flows):
        logging.error("NPV Calculation Error: Dates and cash_flows lengths mismatch.")
        raise ValueError(
            "Dates and cash_flows must have the same length."
        )  # Raise for IRR solver
    if not dates:
        return 0.0
    base = 1.0 + rate
    if base <= 1e-9:
        logging.debug(
            f"NPV Calculation Warning: Base (1+rate) is <= 1e-9 ({base}). Returning NaN."
        )
        return np.nan  # Avoid issues with rate = -1 or less

    start_date = dates[0]
    npv = 0.0
    for i in range(len(cash_flows)):
        try:
            if not isinstance(dates[i], date) or not isinstance(start_date, date):
                logging.debug(f"NPV Calc Error: Invalid date type at index {i}.")
                return np.nan
            time_delta_years = (dates[i] - start_date).days / 365.0
            if not np.isfinite(time_delta_years) or (
                time_delta_years < -1e-9 and i > 0
            ):
                logging.debug(
                    f"NPV Calc Error: Invalid time delta ({time_delta_years}) at index {i}."
                )
                return np.nan
            if not np.isfinite(cash_flows[i]):
                continue  # Skip non-finite flows silently

            if abs(base) < 1e-9 and time_delta_years != 0:
                logging.debug(
                    "NPV Calc Warning: Base is near zero with non-zero time delta. Returning NaN."
                )
                return np.nan

            # Check for negative base with non-integer exponent
            if base < 0 and time_delta_years != int(time_delta_years):
                logging.debug(
                    f"NPV Calc Warning: Negative base ({base}) with non-integer exponent ({time_delta_years}). Returning NaN."
                )
                return np.nan

            denominator = base**time_delta_years
            if not np.isfinite(denominator) or abs(denominator) < 1e-12:
                logging.debug(
                    f"NPV Calc Warning: Invalid denominator ({denominator}) at index {i}. Returning NaN."
                )
                return np.nan

            term_value = cash_flows[i] / denominator
            if not np.isfinite(term_value):
                logging.debug(
                    f"NPV Calc Warning: Non-finite term value ({term_value}) at index {i}. Returning NaN."
                )
                return np.nan
            npv += term_value
        except OverflowError:
            logging.warning(
                f"NPV Calculation OverflowError at index {i} (Rate: {rate}, TimeDelta: {time_delta_years}). Returning NaN."
            )
            return np.nan
        except TypeError as e:
            logging.warning(
                f"NPV Calculation TypeError at index {i}: {e}. Returning NaN."
            )
            return np.nan
        except Exception as e:
            logging.exception(f"Unexpected error in NPV calculation loop at index {i}")
            return np.nan  # Catch any other unexpected calculation errors
    return float(npv) if np.isfinite(npv) else np.nan


def calculate_irr(dates: List[date], cash_flows: List[float]) -> float:
    """
    Calculates the Internal Rate of Return (IRR/MWR) for a series of cash flows.

    Finds the discount rate at which the Net Present Value (NPV) of the cash flows equals zero.
    Uses numerical methods (Newton-Raphson with Brentq fallback) to solve for the rate.
    Requires at least two cash flows, sorted dates, and a valid investment pattern
    (typically starting with a negative flow and having at least one positive flow later).

    Args:
        dates (List[date]): A list of dates corresponding to the cash flows. Must be sorted.
        cash_flows (List[float]): A list of cash flows. Must be the same length as dates.

    Returns:
        float: The calculated Internal Rate of Return (IRR) as a decimal (e.g., 0.1 for 10%),
               or np.nan if the calculation fails, inputs are invalid, dates are unsorted,
               or the cash flow pattern doesn't allow for a standard IRR calculation
               (e.g., all positive flows, all negative flows, first non-zero flow is positive).
    """
    # 1. Basic Input Validation
    if len(dates) < 2 or len(cash_flows) < 2 or len(dates) != len(cash_flows):
        logging.debug("DEBUG IRR: Fail - Length mismatch or < 2")
        return np.nan

    # --- ADDED: Handle zero-duration investments ---
    # If all transactions occur on the same day, the time delta is always zero,
    # making the IRR calculation mathematically undefined or infinite.
    if all(d == dates[0] for d in dates):
        logging.debug("DEBUG IRR: Fail - All cash flows occur on the same date.")
        return np.nan
    # --- END ADDED ---

    if any(
        not isinstance(cf, (int, float)) or not np.isfinite(cf) for cf in cash_flows
    ):
        logging.debug("DEBUG IRR: Fail - Non-finite cash flows")
        return np.nan
    # Check dates are valid and sorted
    try:
        if not all(isinstance(d, date) for d in dates):
            raise TypeError("Not all elements in dates are date objects")
        for i in range(1, len(dates)):
            if dates[i] < dates[i - 1]:
                raise ValueError("Dates are not sorted")
    except (TypeError, ValueError) as e:
        logging.debug(f"DEBUG IRR: Fail - Date validation error: {e}")
        return np.nan

    # 2. Cash Flow Pattern Validation
    # A valid investment for IRR requires an initial outflow (negative) and at least one inflow (positive).
    first_non_zero_flow = None
    first_non_zero_idx = -1
    non_zero_cfs_list = []
    for idx, cf in enumerate(cash_flows):
        if abs(cf) > 1e-9:
            non_zero_cfs_list.append(cf)
            if first_non_zero_flow is None:
                first_non_zero_flow = cf
                first_non_zero_idx = idx

    # Case 1: All cash flows are zero.
    if first_non_zero_flow is None:
        logging.debug("DEBUG IRR: Fail - All flows are zero")
        return np.nan
    # Case 2: The first cash flow must be an investment (negative).
    if first_non_zero_flow >= -1e-9:
        logging.debug(
            f"DEBUG IRR: Fail - First non-zero flow is non-negative: {first_non_zero_flow} in {cash_flows}"
        )
        return np.nan
    # Case 3: There must be at least one positive cash flow (a return).
    has_positive_flow = any(cf > 1e-9 for cf in non_zero_cfs_list)
    if not has_positive_flow:
        logging.debug(f"DEBUG IRR: Fail - No positive flows found: {cash_flows}")
        return np.nan

    # 3. Solver Logic
    irr_result = np.nan
    # First, try the Newton-Raphson method. It's fast but needs a good guess and can fail to converge.
    try:
        irr_result = optimize.newton(
            calculate_npv, x0=0.1, args=(dates, cash_flows), tol=1e-6, maxiter=100
        )
        if (
            not np.isfinite(irr_result) or irr_result <= -1.0 or irr_result > 100.0
        ):  # Check range
            raise RuntimeError("Newton result out of reasonable range")
        npv_check = calculate_npv(irr_result, dates, cash_flows)
        if not np.isclose(
            npv_check, 0.0, atol=1e-4
        ):  # Check if it finds the root accurately
            raise RuntimeError(
                f"Newton result did not produce zero NPV (NPV={npv_check:.4f})"
            )
    except (RuntimeError, OverflowError):
        # If Newton fails, fall back to the more robust Brent's method (brentq).
        # This method requires a bracket [a, b] where the NPV at a and b have opposite signs.
        try:
            lower_bound, upper_bound = -0.9999, 50.0
            npv_low = calculate_npv(lower_bound, dates, cash_flows)
            npv_high = calculate_npv(upper_bound, dates, cash_flows)
            # --- ADDED: More explicit logging for brentq failure ---
            logging.debug(
                f"DEBUG IRR: Newton failed. Brentq bounds NPV: Low={npv_low}, High={npv_high}"
            )
            if (
                pd.notna(npv_low) and pd.notna(npv_high) and npv_low * npv_high < 0
            ):  # Check sign change
                irr_result = optimize.brentq(
                    calculate_npv,
                    a=lower_bound,
                    b=upper_bound,
                    args=(dates, cash_flows),
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100,
                )
                if not np.isfinite(irr_result) or irr_result <= -1.0:
                    irr_result = np.nan
            else:
                irr_result = np.nan
                logging.debug(
                    "DEBUG IRR: Brentq skipped - NPV at bounds do not have opposite signs."
                )
        except (ValueError, RuntimeError, OverflowError, Exception):
            irr_result = np.nan

    # 4. Final Validation and Return
    if not (
        isinstance(irr_result, (float, int))
        and np.isfinite(irr_result)
        and irr_result > -1.0
    ):
        logging.debug(f"DEBUG IRR: Fail - Final result invalid: {irr_result}")
        return np.nan
    return irr_result


# --- Cash Flow Helpers ---
def get_cash_flows_for_symbol_account(
    symbol: str,
    account: str,
    transactions: pd.DataFrame,
    final_market_value_local: float,
    end_date: date,
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
    symbol_account_tx_filtered = transactions[
        (transactions["Symbol"] == symbol) & (transactions["Account"] == account)
    ]
    if symbol_account_tx_filtered.empty:
        return [], []
    symbol_account_tx = symbol_account_tx_filtered.copy()
    dates_flows = defaultdict(float)
    symbol_account_tx.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )

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
        tx_date = row["Date"].date()
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        if tx_type == "buy" or tx_type == "deposit":
            if pd.notna(qty) and qty > 0 and pd.notna(price_local):
                cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == "sell" or tx_type == "withdrawal":
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = (qty_abs * price_local) - commission_local
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
    if not sorted_dates and abs(final_market_value_local) < 1e-9:
        return [], []
    final_dates = list(sorted_dates)
    final_flows = [float(dates_flows[d]) for d in final_dates]
    final_market_value_local_abs = (
        abs(final_market_value_local) if pd.notna(final_market_value_local) else 0.0
    )
    if final_market_value_local_abs > 1e-9 and isinstance(end_date, date):
        if not final_dates:
            first_tx_date_for_holding = (
                symbol_account_tx["Date"].min().date()
                if not symbol_account_tx.empty
                else end_date
            )
            if end_date >= first_tx_date_for_holding:
                final_dates.append(end_date)
                final_flows.append(final_market_value_local_abs)
            else:
                return [], []
        elif end_date >= final_dates[-1]:
            if final_dates[-1] == end_date:
                final_flows[-1] += final_market_value_local_abs
            else:
                final_dates.append(end_date)
                final_flows.append(final_market_value_local_abs)

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
) -> Tuple[List[date], List[float]]:
    """
    Calculates cash flows for Money-Weighted Return (MWR) for a specific account in the target currency.

    Processes transactions for a single account, calculating the cash flow impact of each
    transaction (buys, sells, dividends, fees, cash deposits/withdrawals) in its local currency.
    Converts these local currency flows to the `target_currency` using the provided `fx_rates`.
    The final account market value (already in `target_currency`) is added as the terminal flow.
    Note: The sign convention for MWR cash flows is often flipped compared to IRR (deposits/buys
    are positive, withdrawals/sells are negative) before solving. This function applies that flip.

    Args:
        account_transactions (pd.DataFrame): The transactions DataFrame filtered for a specific account.
                                             Must contain 'Date', 'Symbol', 'Type', 'Quantity',
                                             'Price/Share', 'Commission', 'Total Amount', and
                                             'Local Currency' columns.
        final_account_market_value (float): The final market value of the entire account in the
                                            `target_currency` as of the end_date.
        end_date (date): The end date for the calculation period.
        target_currency (str): The target currency for the MWR calculation.
        fx_rates (Optional[Dict[str, float]]): A dictionary of FX rates relative to a base currency
                                               (typically USD, e.g., {'JPY': 150.0, 'EUR': 0.9})
                                               used for currency conversion via `get_conversion_rate`.
        display_currency (str): The display currency used in log messages (informational only).

    Returns:
        Tuple[List[date], List[float]]: A tuple containing:
            - List[date]: A list of dates for the cash flows, sorted chronologically.
            - List[float]: A list of cash flows in the `target_currency`, with signs flipped
                           for MWR calculation (deposits/buys positive, withdrawals/sells negative).
                           The final market value is added as a positive flow on the end_date.
            Returns ([], []) if no relevant transactions or final value exist, or if the
            cash flow pattern is invalid for IRR/MWR.
    """
    if account_transactions.empty:
        return [], []
    acc_tx_copy = account_transactions.copy()
    dates_flows = defaultdict(float)
    acc_tx_copy.sort_values(by=["Date", "original_index"], inplace=True, ascending=True)

    for _, row in acc_tx_copy.iterrows():
        tx_type = str(row.get("Type", "")).lower().strip()
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
        local_currency = row["Local Currency"]
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        if symbol != CASH_SYMBOL_CSV:
            if tx_type == "buy":
                if pd.notna(qty) and qty > 0 and pd.notna(price_local):
                    cash_flow_local = -(
                        (qty_abs * price_local) + commission_local
                    )  # OUT (-)
            elif tx_type == "sell":
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = (
                        qty_abs * price_local
                    ) - commission_local  # IN (+)
            elif tx_type == "short sell" and symbol in SHORTABLE_SYMBOLS:
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = (
                        qty_abs * price_local
                    ) - commission_local  # IN (+)
            elif tx_type == "buy to cover" and symbol in SHORTABLE_SYMBOLS:
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = -(
                        (qty_abs * price_local) + commission_local
                    )  # OUT (-)
            elif tx_type == "dividend":
                dividend_amount_local_cf = 0.0
                if pd.notna(total_amount_local) and total_amount_local != 0:
                    dividend_amount_local_cf = total_amount_local
                elif pd.notna(price_local) and price_local != 0:
                    dividend_amount_local_cf = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                cash_flow_local = dividend_amount_local_cf - commission_local  # IN (+)
            elif tx_type == "fees":
                if pd.notna(commission_local):
                    cash_flow_local = -(abs(commission_local))  # OUT (-)
            elif tx_type in ["split", "stock split"]:
                cash_flow_local = 0.0
                if pd.notna(commission_local) and commission_local != 0:
                    cash_flow_local = -abs(commission_local)  # OUT (-)
        elif symbol == CASH_SYMBOL_CSV:
            if tx_type == "deposit" or tx_type == "buy":
                if pd.notna(qty):
                    cash_flow_local = abs(qty)  # IN (+)
                cash_flow_local -= commission_local
            elif tx_type == "withdrawal" or tx_type == "sell":
                if pd.notna(qty):
                    cash_flow_local = -abs(qty)  # OUT (-)
                cash_flow_local -= commission_local
            elif tx_type == "dividend":
                dividend_amount_local_cf = 0.0
                if pd.notna(total_amount_local) and total_amount_local != 0:
                    dividend_amount_local_cf = total_amount_local
                elif pd.notna(price_local) and price_local != 0:
                    dividend_amount_local_cf = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                cash_flow_local = dividend_amount_local_cf - commission_local  # IN (+)
            elif tx_type == "fees":
                if pd.notna(commission_local):
                    cash_flow_local = -abs(commission_local)  # OUT (-)

        cash_flow_target = cash_flow_local
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            if local_currency != target_currency:
                rate = get_conversion_rate(local_currency, target_currency, fx_rates)
                if pd.isna(rate):  # Check for NaN explicitly
                    logging.warning(
                        f"Warning: MWR calc cannot convert flow on {tx_date} from {local_currency} to {target_currency} (FX rate missing/invalid). Skipping flow."
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
    final_flows = [-dates_flows[d] for d in final_dates]  # <<< SIGN FLIP
    final_market_value_target = (
        float(final_account_market_value)
        if pd.notna(final_account_market_value)
        else 0.0
    )
    final_market_value_abs = abs(final_market_value_target)

    if final_market_value_abs > 1e-9 and isinstance(end_date, date):
        if not final_dates:
            first_tx_date_for_account = (
                acc_tx_copy["Date"].min().date() if not acc_tx_copy.empty else end_date
            )
            if end_date >= first_tx_date_for_account:
                # If no flows but final value exists, need initial flow (usually 0)
                # This case is complex for MWR, maybe return empty?
                # For now, let's return empty if no flow dates exist.
                return [], []
            else:
                return [], []  # End date is before first transaction
        elif end_date >= final_dates[-1]:
            if final_dates[-1] == end_date:
                final_flows[-1] += final_market_value_abs  # Add to last flow
            else:
                final_dates.append(end_date)
                final_flows.append(final_market_value_abs)  # Append as new flow

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
            f"Warning: get_conversion_rate received invalid fx_rates type. Returning NaN"
        )
        return np.nan  # Return NaN on invalid rates dict

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()
    rate_A_per_USD = fx_rates.get(from_curr_upper)
    # Added debug log for lookup
    logging.debug(
        f"get_conversion_rate: Looking up {from_curr_upper}/USD (key '{from_curr_upper}'): {rate_A_per_USD}"
    )
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0
    rate_B_per_USD = fx_rates.get(to_curr_upper)
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
def get_historical_price(
    symbol_key: str, target_date: date, prices_dict: Dict[str, pd.DataFrame]
) -> Optional[float]:
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
        Optional[float]: The price for the symbol on the target date (or the last available price
                         before it). Returns None if the symbol is not found, the date is before
                         the first available price point, or an error occurs during lookup.
    """
    if symbol_key not in prices_dict or prices_dict[symbol_key].empty:
        return None
    df = prices_dict[symbol_key]
    try:
        if not all(isinstance(idx, date) for idx in df.index):
            df.index = pd.to_datetime(df.index).date
        if not isinstance(target_date, date):
            return None  # Return None on invalid target date type
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
        combined_index = df.index.union([target_date])
        df_reindexed = df.reindex(combined_index, method="ffill")
        price = df_reindexed.loc[target_date]["price"]
        if pd.isna(price) and target_date < df.index.min():
            return None
        return float(price) if pd.notna(price) else None
    except KeyError:
        return None
    except Exception as e:
        logging.error(
            f"ERROR getting historical price for {symbol_key} on {target_date}: {e}"
        )
        return None


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
        logging.warning(f"Hist FX Bridge: Invalid historical_fx_data type received.")
        return np.nan  # Return NaN on invalid input dict

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()
    rate_A_per_USD = np.nan
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0
    else:
        pair_A = f"{from_curr_upper}=X"
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
        pair_B = f"{to_curr_upper}=X"
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
    if (
        normalized_symbol == CASH_SYMBOL_CSV
        or normalized_symbol
        in user_excluded_symbols  # CORRECTLY Use user-defined exclusions
    ):
        logging.debug(
            f"  Symbol '{normalized_symbol}' is CASH or EXCLUDED. Returning None."
        )
        return None

    # --- 2. Check Explicit Map (if not excluded) ---
    # Ensure keys in the map are also normalized for comparison
    normalized_map = {
        k.upper().strip(): v.upper().strip() for k, v in user_symbol_map.items()
    }  # Use user-defined map
    if normalized_symbol in normalized_map:
        mapped_symbol = normalized_map[normalized_symbol]
        logging.debug(
            f"  Found in explicit map: '{normalized_symbol}' -> '{mapped_symbol}'"
        )
        return normalized_map[normalized_symbol]

    # --- 3. Apply Automatic Conversion Rules (if not found in map) ---
    # Handle Thai stocks (:BKK -> .BK)
    logging.debug(
        f"  '{normalized_symbol}' not in explicit map. Checking rules..."
    )  # ADDED Logging
    if normalized_symbol.endswith(":BKK"):
        base_symbol = normalized_symbol[:-4]
        # Check if the base symbol itself has an explicit mapping (e.g., "BRK.B" -> "BRK-B")
        # This check is now redundant because we check the full normalized_symbol first
        # if base_symbol in normalized_map:
        #     base_symbol = normalized_map[base_symbol]
        if "." in base_symbol or len(base_symbol) == 0:
            logging.warning(
                f"Hist WARN: Skipping potentially invalid BKK conversion: {internal_symbol}"
            )
            return None
        # --- ADDED Logging ---
        converted_symbol = f"{base_symbol.upper()}.BK"
        logging.debug(
            f"  Applied :BKK rule: '{normalized_symbol}' -> '{converted_symbol}'"
        )
        # --- END Logging ---
        return f"{base_symbol.upper()}.BK"

    # --- 4. Check for other invalid formats ---
    if " " in normalized_symbol or any(c in normalized_symbol for c in [":", ","]):
        logging.warning(
            f"map_to_yf_symbol WARN: Skipping potentially invalid symbol format for YF: {internal_symbol}"  # Modified log source
        )
        return None

    # --- 5. Return normalized symbol if no rules applied ---
    # --- ADDED Logging ---
    logging.debug(
        f"  No rules applied. Returning normalized symbol: '{normalized_symbol.upper()}'"
    )
    # --- END Logging ---
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
            return f"{currency_symbol}{val_float/1e12:,.{decimals}f} T"
        elif val_float >= 1e9:
            return f"{currency_symbol}{val_float/1e9:,.{decimals}f} B"
        elif val_float >= 1e6:
            return f"{currency_symbol}{val_float/1e6:,.{decimals}f} M"
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
