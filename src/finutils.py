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
from datetime import date, datetime
import hashlib
import os
import config
import math
import logging
from scipy import optimize
import warnings
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
    hasher = hashlib.sha256()
    
    # Helper to hash a single file into the main hasher
    def _update_hash_with_file(path: str):
        try:
            with open(path, "rb") as file:
                while chunk := file.read(HASH_CHUNK_SIZE):
                    hasher.update(chunk)
            return True
        except FileNotFoundError:
            return False # Ignore if disappeared (e.g. checkpointed)


    try:
        # 1. Hash the main file
        if not _update_hash_with_file(filepath):
            # Main file not found or error
            logging.warning(f"Warning: Main file not found for hashing: {filepath}")
            return HASH_ERROR_NOT_FOUND

        # 2. Check for SQLite companion files if applicable
        # (Naive check based on extension, strictly redundant but safe)
        lower_path = filepath.lower()
        if lower_path.endswith((".db", ".sqlite", ".sqlite3")):
            wal_path = filepath + "-wal"
            shm_path = filepath + "-shm"
            
            if os.path.exists(wal_path):
                 _update_hash_with_file(wal_path)
                 # logging.debug(f"Hashed WAL file: {wal_path}")
            
            if os.path.exists(shm_path):
                 _update_hash_with_file(shm_path)
                 # logging.debug(f"Hashed SHM file: {shm_path}")

        return hasher.hexdigest()

    except PermissionError:
        logging.error(f"Permission denied accessing file {filepath} for hashing.")
        return HASH_ERROR_PERMISSION
    except IOError as e:
        logging.error(f"I/O error hashing file {filepath}: {e}")
        return HASH_ERROR_IO
    except Exception:
        logging.exception(f"Unexpected error hashing file {filepath}")
        return HASH_ERROR_UNEXPECTED


# --- ADDED: Currency Symbol Helper ---
def get_currency_symbol_from_code(currency_code: str) -> str:
    """
    Returns the currency symbol for a given currency code.
    Falls back to the code itself if no symbol is mapped.
    """
    if not currency_code or not isinstance(currency_code, str):
        return ""
    return CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code.upper())


# --- END ADDED ---


# --- Indicated Dividend Helper ---
def get_dividend_details(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates dividend details: frequency_months and indicated_annual_rate.
    Unifies logic between summary metrics and projected income charts.
    """
    if not info:
        return {"frequency_months": 3, "indicated_annual_rate": 0.0}
        
    last_div_val = float(info.get("lastDividendValue", 0.0) or 0.0)
    
    try:
        div_rate = float(info.get("dividendRate", 0.0) or 0.0)
    except (ValueError, TypeError):
        div_rate = 0.0
        
    try:
        trailing_rate = float(info.get("trailingAnnualDividendRate", 0.0) or 0.0)
    except (ValueError, TypeError):
        trailing_rate = 0.0
    
    if div_rate <= 0: div_rate = trailing_rate if trailing_rate else 0.0
    
    if last_div_val <= 0 and div_rate <= 0:
        return {"frequency_months": 3, "indicated_annual_rate": 0.0}
        
    freq = 4
    consistent = False
    if last_div_val > 0 and div_rate > 0:
        ratio = div_rate / last_div_val
        # Map the annual-rate / last-paid ratio to a payment cadence. A match
        # also means the two fields agree; no match means they disagree (e.g. a
        # recent dividend change) and we keep the default quarterly cadence.
        for lo, hi, f in ((10.0, 14.5, 12), (3.4, 5.8, 4), (1.4, 2.6, 2), (0.7, 1.3, 1)):
            if lo <= ratio <= hi:
                freq = f
                consistent = True
                break

    # Indicated (forward-looking) annual rate.
    #   - When `dividendRate` (Yahoo's forward annual figure) and the last paid
    #     dividend agree, either basis works; use lastDividendValue × freq so a
    #     just-announced raise still shows once `dividendRate` lags.
    #   - When they DISAGREE, trust `dividendRate`: it updates as soon as a new
    #     dividend is declared, whereas `lastDividendValue` reflects only the
    #     last *paid* dividend and lags an increase/cut. e.g. NVDA raised its
    #     quarterly from $0.01 to $0.25 (dividendRate=1.0 captures the $1.00/yr
    #     run-rate; lastDividendValue=0.01 is stale).
    if div_rate > 0 and (not consistent or last_div_val <= 0):
        indicated_rate = float(div_rate)
    elif last_div_val > 0:
        indicated_rate = float(last_div_val * freq)
    else:
        indicated_rate = float(div_rate)
    freq_months = 12 // freq
    
    return {
        "frequency_months": freq_months,
        "indicated_annual_rate": indicated_rate,
        "last_dividend_value": last_div_val
    }

def calculate_indicated_dividend(info: Dict[str, Any]) -> float:
    """Convenience wrapper for backward compatibility."""
    return get_dividend_details(info)["indicated_annual_rate"]


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
    if not info:
        return None

    # Funds/ETFs sometimes carry an explicit 'yield' fraction.
    explicit = info.get("yield")
    if explicit is not None:
        try:
            v = float(explicit)
            if v > 0:
                return v if v < 1.0 else v / 100.0
        except (TypeError, ValueError):
            pass

    # 1. Indicated annual rate / current price (most reliable).
    indicated = get_dividend_details(info).get("indicated_annual_rate") or 0.0
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = 0.0
    if indicated > 0 and price > 0:
        return indicated / price

    # 2. Trailing annual dividend yield (already a fraction).
    try:
        ty = float(info.get("trailingAnnualDividendYield"))
        if ty > 0:
            return ty
    except (TypeError, ValueError):
        pass

    # 3. Last resort: raw Yahoo dividendYield, guessing fraction vs percent.
    try:
        rv = float(info.get("dividendYield"))
        if rv > 0:
            return rv / 100.0 if rv > 0.30 else rv
    except (TypeError, ValueError):
        pass

    return None


# --- Cash Symbol Helpers ---

def is_cash_symbol(symbol: str) -> bool:
    """Checks if a symbol is a cash symbol (e.g., '$CASH', 'Cash ($)')."""
    if not isinstance(symbol, str):
        return False
    symbol_lower = symbol.lower()
    return symbol_lower.startswith(CASH_SYMBOL_CSV.lower()) or symbol_lower.startswith(
        "cash ("
    )


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
    if not isinstance(symbol, str) or symbol.upper() != CASH_SYMBOL_CSV.upper():
        return False
    if not isinstance(tx_type, str):
        return False
    return tx_type.strip().lower() in ("deposit", "withdrawal")


def compute_external_flow_mask(df: "pd.DataFrame") -> "pd.Series":
    """Vectorized version of `is_external_flow_row` for a DataFrame.

    Returns a boolean pd.Series aligned with df.index. Expects df to have
    'Symbol' and 'Type' columns. 'Note' is not consulted.
    """
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)

    symbol_upper = df["Symbol"].astype(str).str.upper()
    type_lower = df["Type"].astype(str).str.lower().str.strip()

    return (
        (symbol_upper == CASH_SYMBOL_CSV.upper())
        & (type_lower.isin(["deposit", "withdrawal"]))
    )


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
        raise ValueError("The lengths of dates and cash_flows do not match.")
    if len(dates) == 0:
        return 0.0
    base = 1.0 + rate
    if base <= 1e-18:
        # logging.debug(
        #    f"NPV Calculation Warning: Base (1+rate) is <= 1e-18 ({base}). Returning NaN."
        # )
        return np.nan  # Avoid issues with rate = -1 or less

    start_date = dates[0]
    npv = 0.0
    for i in range(len(cash_flows)):
        try:
            time_delta_years = (dates[i] - start_date).days / 365.0
            if time_delta_years < 0:
                logging.warning(
                    f"NPV Calc Error: Invalid time delta ({time_delta_years}) at index {i}."
                )
                return np.nan
            if not np.isfinite(cash_flows[i]):
                continue  # Skip non-finite flows silently

            # Check for negative base with non-integer exponent
            if base < 0 and time_delta_years != int(time_delta_years):
                return np.nan

            # Safe Exponentiation
            try:
                # We want to calculate denominator = base**time_delta_years
                # ln(1.8e308) ~= 709.7
                log_val = math.log(base) * time_delta_years
                if log_val > 700.0:
                    denominator = np.inf
                elif log_val < -700.0:
                    denominator = 0.0
                else:
                    denominator = base**time_delta_years
            except (ValueError, OverflowError):
                 denominator = np.nan

            if denominator == 0.0:
                # If denominator is zero, term is +/- infinity (magnified at rate -> -1)
                if cash_flows[i] > 0:
                    npv += 1e150 
                elif cash_flows[i] < 0:
                    npv -= 1e150
                continue
                
            if not np.isfinite(denominator) or np.isnan(denominator):
                # If denominator is infinite (huge rate), term is effectively 0
                # unless it's the first term (t=0)
                if time_delta_years == 0:
                    npv += cash_flows[i]
                continue

            npv += cash_flows[i] / denominator
        except Exception:
            return np.nan
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
    # 2. Strip leading and trailing zeros
    # Leading zeros change the start date, trailing zeros are just noise.
    first_idx = 0
    while first_idx < len(cash_flows) and abs(cash_flows[first_idx]) < 1e-9:
        first_idx += 1
    
    last_idx = len(cash_flows) - 1
    while last_idx > first_idx and abs(cash_flows[last_idx]) < 1e-9:
        last_idx -= 1
        
    if last_idx - first_idx < 1:
        return np.nan
        
    dates = dates[first_idx : last_idx + 1]
    cash_flows = cash_flows[first_idx : last_idx + 1]

    # 3. Cash Flow Pattern Validation
    # A valid investment for IRR requires an initial outflow (negative) and at least one inflow (positive).
    first_non_zero_flow = None
    first_non_zero_idx = -1
    non_zero_cfs_list = []
    max_abs_flow = 0.0  # Tracks max magnitude for normalization
    
    for idx, cf in enumerate(cash_flows):
        abs_cf = abs(cf)
        if abs_cf > 1e-9:
            non_zero_cfs_list.append(cf)
            if abs_cf > max_abs_flow:
                max_abs_flow = abs_cf
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

    # --- ADDED: Normalize Cash Flows ---
    # Solver can fail with very large numbers (e.g. JPY, VND).
    # Scaling flows does not change IRR.
    # We scale down by max_abs_flow to keep numbers around [-1, 1].
    
    solver_flows = cash_flows
    if max_abs_flow > 1e6: # Only scale if numbers are somewhat large (1M+)
        scale_factor = max_abs_flow
        solver_flows = [cf / scale_factor for cf in cash_flows]
        # logging.debug(f"DEBUG IRR: Scaled flows by {scale_factor}. First Flow: {solver_flows[0]}")
    # --- END ADDED ---

    # 3. Solver Logic
    with warnings.catch_warnings():
        # Ignore common numerical warnings from scipy's root-finding algorithms.
        # These often occur with unusual cash flows but may still produce a valid result.
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in scalar divide",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )

        irr_result = np.nan
        # 1. Try Newton-Raphson with multiple initial guesses
        try:
            for x0 in [0.1, 0.0, -0.2, 0.5, -0.5]:
                try:
                    res = optimize.newton(
                        calculate_npv, x0=x0, args=(dates, solver_flows), tol=1e-6, maxiter=50
                    )
                    if np.isfinite(res) and res > -1.0 and res < 1000.0:
                        npv_check = calculate_npv(res, dates, solver_flows)
                        if np.isclose(npv_check, 0.0, atol=1e-2):
                            irr_result = res
                            break
                except:
                    continue
        except:
            pass

        # 2. If Newton fails, fallback to Brentq with expanding brackets
        if np.isnan(irr_result):
            for lb in [-0.9, -0.99, -0.999, -0.99999, -0.99999999]:
                try:
                    upper_bound = 100000.0
                    npv_low = calculate_npv(lb, dates, solver_flows)
                    npv_high = calculate_npv(upper_bound, dates, solver_flows)
                    
                    if pd.notna(npv_low) and pd.notna(npv_high) and npv_low * npv_high < 0:
                        res = optimize.brentq(
                            calculate_npv,
                            a=lb,
                            b=upper_bound,
                            args=(dates, solver_flows),
                            xtol=1e-6,
                            rtol=1e-6,
                            maxiter=100,
                        )
                        if np.isfinite(res) and res > -1.0:
                            irr_result = res
                            break
                except:
                    continue

    # 4. Final Validation and Return
    if not (
        isinstance(irr_result, (float, int))
        and np.isfinite(irr_result)
        and irr_result > -1.0
    ):
        # Format flows for readability
        readable_flows = [f"{d.strftime('%Y-%m-%d')}: {f:,.2f}" for d, f in zip(dates, solver_flows)]
        # Truncate if too long
        if len(readable_flows) > 100:
            flow_str = f"[{', '.join(readable_flows[:10])} ... {', '.join(readable_flows[-10:])}] ({len(readable_flows)} flows)"
        else:
            flow_str = f"[{', '.join(readable_flows)}]"
            
             
        logging.warning(f"IRR Calculation Failed. Flows used: {flow_str}")
        return np.nan
        
    # --- DIAGNOSTIC LOGGING for extreme results ---
    if irr_result > 10.0: # > 1000% p.a.
        logging.warning(f"Extreme IRR detected ({irr_result*100:.1f}%). Investigating flows...")
        # Log top 10 largest absolute flows for debugging
        top_flows = sorted(list(zip(dates, solver_flows)), key=lambda x: abs(x[1]), reverse=True)[:10]
        readable_top = [f"{d.strftime('%Y-%m-%d')}: {f:,.2f}" for d, f in top_flows]
        logging.warning(f"Top Flows contributing to IRR: [{', '.join(readable_top)}]")
        
    return irr_result


# --- Cash Flow Helpers ---
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
        account_mask = (
            (transactions_df["Account"] == account)
            | (transactions_df["To Account"] == account)
        )
        # --- ADDED: Check Note for account name (handles legacy transfers where Account col is misaligned) ---
        if "Note" in transactions_df.columns:
             account_mask |= (transactions_df["Note"].str.contains(account, case=False, na=False) & (transactions_df["Type"].str.lower() == "transfer"))
        
        symbol_account_tx_filtered = transactions_df[
            (transactions_df["Symbol"] == symbol) & account_mask
        ]
    else:
        # If 'To Account' is missing, fallback to filtering only by 'Account'
        account_mask = (transactions_df["Account"] == account)
        if "Note" in transactions_df.columns:
             account_mask |= (transactions_df["Note"].str.contains(account, case=False, na=False) & (transactions_df["Type"].str.lower() == "transfer"))
             
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

        total_amount_abs = abs(total_amount_local) if pd.notna(total_amount_local) else 0.0
        qty_val_abs = abs(qty) if pd.notna(qty) else 0.0
        cash_amt = total_amount_abs if total_amount_abs > 1e-9 else qty_val_abs

        if symbol == CASH_SYMBOL_CSV:
            if tx_type == "buy" or tx_type == "deposit":
                cash_flow_local = -(cash_amt + commission_local)
            elif tx_type == "sell" or tx_type == "withdrawal":
                cash_flow_local = (cash_amt - commission_local)
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
                if (pd.isna(price_to_use) or price_to_use <= 1e-9) and last_seen_price is not None:
                    price_to_use = last_seen_price
                
                flow = qty * price_to_use if pd.notna(price_to_use) else 0.0
            
            # Determine direction if Account matches but To Account doesn't, or vice-versa, or Note matches
            acct_col = str(row.get("Account", "")).strip()
            note = str(row.get("Note", "")).lower()
            
            # Refined direction detection
            is_from_note = (f"from {account.lower()}" in note)
            is_to_note = (f"to {account.lower()}" in note)
            
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
    historical_fx_rates: Optional[Dict[Tuple[date, str], float]] = None, # ADDED: Historical FX Data
    include_accounts: Optional[List[str]] = None, # ADDED: To determine transfer direction
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
        portfolio_basis = (flow_basis == "portfolio")

        if symbol != CASH_SYMBOL_CSV:
            if portfolio_basis:
                if tx_type == "transfer":
                    is_outbound = False
                    is_inbound = False
                    acct = str(row.get("Account", "")).strip().upper()
                    to_acct = str(row.get("To Account", "")).strip().upper()

                    if acct in included_set: is_outbound = True
                    if to_acct and to_acct in included_set: is_inbound = True

                    if is_outbound and not is_inbound:
                        if pd.notna(qty) and pd.notna(price_local):
                            cash_flow_local = (abs(qty) * price_local) - abs(commission_local)
                    elif not is_outbound and is_inbound:
                        if pd.notna(qty) and pd.notna(price_local):
                            cash_flow_local = -(abs(qty) * price_local)
                # buy/sell/dividend/short sell/buy to cover/fees/tax/interest: 0.0 (internal)
            elif tx_type == "buy" or tx_type == "deposit":
                # External asset contribution or purchase
                if pd.notna(qty) and pd.notna(price_local):
                    cash_flow_local = -( (qty_abs * price_local) + commission_local ) # OUT from pocket (-)
            elif tx_type == "sell" or tx_type == "withdrawal":
                # External asset removal or sale
                if pd.notna(qty) and pd.notna(price_local):
                    cash_flow_local = (qty_abs * price_local) - commission_local # IN to pocket (+)
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
                cash_flow_local = -abs(commission_local)  # BUG-08 FIX: Use abs() consistently
            elif tx_type == "transfer":
                # Asset Transfer logic
                is_outbound = False
                is_inbound = False
                acct = str(row.get("Account", "")).strip().upper()
                to_acct = str(row.get("To Account", "")).strip().upper()

                if acct in included_set: is_outbound = True
                if to_acct and to_acct in included_set: is_inbound = True

                if is_outbound and not is_inbound:
                    # Asset leaving scope -> Withdrawal -> Positive MWR Flow
                    if pd.notna(qty) and pd.notna(price_local):
                        cash_flow_local = (abs(qty) * price_local) - abs(commission_local)  # BUG-09 FIX: Subtract commission (broker takes it)
                elif not is_outbound and is_inbound:
                    # Asset entering scope -> Deposit -> Negative MWR Flow
                    if pd.notna(qty) and pd.notna(price_local):
                        cash_flow_local = -(abs(qty) * price_local)

        elif symbol == CASH_SYMBOL_CSV:
            total_amount_abs = abs(total_amount_local) if pd.notna(total_amount_local) else 0.0
            qty_val_abs = abs(qty) if pd.notna(qty) else 0.0
            cash_amt = total_amount_abs if total_amount_abs > 1e-9 else qty_val_abs

            if portfolio_basis:
                # Portfolio-level MWR: use the centralized external-flow
                # classifier (see is_external_flow_row above). This means:
                #   - $CASH Deposit/Withdrawal with non-Auto-generated Note → flow
                #   - $CASH buy/sell (kitmatan-style settlement) → 0
                #   - $CASH Deposit/Withdrawal with Auto-generated: Note → 0
                #     (dheematan-style synthetic per-trade entries)
                #   - $CASH dividend/interest → 0 (internal income)
                # Same definition as the TWR engine — see
                # _calculate_daily_net_cash_flow_vectorized in portfolio_logic.py.
                if is_external_flow_row(symbol, tx_type, row.get("Note", "")):
                    if tx_type == "deposit":
                        cash_flow_local = -(cash_amt + commission_local)
                    elif tx_type == "withdrawal":
                        cash_flow_local = (cash_amt - commission_local)
                # else: 0 (internal trade rotation, synthetic, or income on cash)
            elif tx_type in ["deposit", "buy"]:
                # External cash deposit (legacy per_trade)
                cash_flow_local = -(cash_amt + commission_local) # OUT from pocket (-)
            elif tx_type in ["withdrawal", "sell"]:
                # External cash withdrawal (legacy per_trade)
                cash_flow_local = (cash_amt - commission_local) # IN to pocket (+)
            elif tx_type in ["dividend", "interest"]:
                # Internal interest/dividends on cash are NOT external flows
                cash_flow_local = 0.0
            elif tx_type == "transfer":
                # Cash Transfer logic
                is_outbound = False
                is_inbound = False
                acct = str(row.get("Account", "")).strip().upper()
                to_acct = str(row.get("To Account", "")).strip().upper()

                if acct in included_set: is_outbound = True
                if to_acct and to_acct in included_set: is_inbound = True
                
                if is_outbound and not is_inbound:
                    # Money leaving the scope -> Withdrawal -> Positive MWR Flow
                    cash_flow_local = (cash_amt - commission_local)
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
                    rate = get_conversion_rate(local_currency, target_currency, fx_rates)
                
                if pd.isna(rate):
                    logging.warning(f"MWR calc conversion failed for {local_currency} on {tx_date}. Skipping.")
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
    if target_date is None or not isinstance(target_date, (date, datetime, pd.Timestamp)):
        return np.nan
    
    if symbol_key not in prices_dict or prices_dict[symbol_key].empty:
        # logging.debug(f"get_historical_price: {symbol_key} missing or empty.")
        return np.nan
    
    # Avoid modifying the original DataFrame in prices_dict (it might be shared)
    df = prices_dict[symbol_key]
    
    # DEBUG: Trace THB calls - REMOVED

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
             elif len(df.index) > 0 and isinstance(df.index[0], date) and not isinstance(df.index[0], datetime):
                 # index is likely date objects, asof(Timestamp) might fail.
                 # Convert target_ts back to date for lookup if index is date objects.
                 target_ts = target_ts.date()
                       
             res = df.asof(target_ts)
            

            
        except (TypeError, ValueError) as e_asof:
             logging.debug(f"AsOf initial fail for {symbol_key} on {target_date}: {e_asof}")
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
             price = res.get('price')
             if price is None or pd.isna(price):
                 price = res.get('Adj Close')
             if price is None or pd.isna(price):
                 price = res.get('Close')
        elif isinstance(res, pd.DataFrame): 
             # Duplicate index case
             if not res.empty:
                 if 'price' in res.columns:
                     price = res['price'].iloc[-1]
                 elif 'Adj Close' in res.columns:
                     price = res['Adj Close'].iloc[-1]
                 elif 'Close' in res.columns:
                     price = res['Close'].iloc[-1]
        elif pd.notna(res):
             pass # Scalar?

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
                     col = 'price'
                     if 'price' not in df.columns:
                         if 'Adj Close' in df.columns: col = 'Adj Close'
                         elif 'Close' in df.columns: col = 'Close'
                     
                     backfill_price = df[col].iloc[0]
                     if pd.notna(backfill_price):
                         # logging.debug(f"BACKFILL: {symbol_key} using {backfill_price} for {target_date}")
                         return float(backfill_price)
             except Exception as e_comp:
                 logging.debug(f"Comparison fail in get_historical_price: {e_comp}")

        return float(price) if pd.notna(price) else np.nan

    except Exception as e:
        logging.error(f"ERROR getting historical price for {symbol_key} on {target_date}: {e}")
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
        logging.debug(f"  map_to_yf_symbol: Symbol '{normalized_symbol}' is in EXCLUSION list. Returning None.")
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
                logging.debug(f"  Applied DOT-to-DASH heuristic: '{normalized_symbol}' -> '{converted}'")
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
