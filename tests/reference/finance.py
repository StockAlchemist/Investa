"""Frozen reference for the Rust IRR/NPV in rust/investa_core/src/irr.rs.

finutils.calculate_npv and finutils.calculate_irr exactly as they ran before
the port (commit cd78d447), with the SciPy solvers they call. The Rust port
reproduces them bit for bit; tests/test_rust_finance_parity.py holds it to that.

Never edit these to follow a change in Rust. A deliberate behaviour change
lands in both, with the parity test updated in the same commit.
"""

# ruff: noqa
import logging
import math
import warnings
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from scipy import optimize


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
    _first_non_zero_idx = -1
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
                _first_non_zero_idx = idx

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
    # Case 4: Positive flows are negligible vs. invested capital -> IRR is
    # effectively -100% and no root exists above -1, so the solver cannot
    # succeed. Common for pass-through cash buckets (deposits immediately
    # spent on purchases, leaving a residual balance of a few cents).
    sum_pos = sum(cf for cf in non_zero_cfs_list if cf > 0)
    sum_neg = -sum(cf for cf in non_zero_cfs_list if cf < 0)
    if sum_neg > 0 and sum_pos / sum_neg < 1e-6:
        logging.debug(
            f"DEBUG IRR: Effective total loss (returned {sum_pos:.4f} of {sum_neg:.2f}); skipping solver."
        )
        return np.nan

    # --- ADDED: Normalize Cash Flows ---
    # Solver can fail with very large numbers (e.g. JPY, VND).
    # Scaling flows does not change IRR.
    # We scale down by max_abs_flow to keep numbers around [-1, 1].

    solver_flows = cash_flows
    if max_abs_flow > 1e6:  # Only scale if numbers are somewhat large (1M+)
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
                        calculate_npv,
                        x0=x0,
                        args=(dates, solver_flows),
                        tol=1e-6,
                        maxiter=50,
                    )
                    if np.isfinite(res) and res > -1.0 and res < 1000.0:
                        npv_check = calculate_npv(res, dates, solver_flows)
                        if np.isclose(npv_check, 0.0, atol=1e-2):
                            irr_result = res
                            break
                except Exception:
                    continue
        except Exception:
            pass

        # 2. If Newton fails, scan the NPV curve across the feasible rate range
        # to bracket a root, then solve with Brentq. Unlike fixed brackets,
        # this also finds roots where NPV has the same sign at both extremes
        # (paired roots), and it tells us definitively when NO root exists.
        found_bracket = False
        if np.isnan(irr_result):
            scan_grid = np.concatenate(
                [
                    -1.0 + np.logspace(-8, 0, 40),  # -0.99999999 .. 0.0
                    np.linspace(0.1, 10.0, 34)[1:],  # 0.4 .. 10
                    np.logspace(1.05, 5, 15),  # ~11 .. 100000
                ]
            )
            prev_r = None
            prev_npv = None
            for r_scan in scan_grid:
                try:
                    npv = calculate_npv(r_scan, dates, solver_flows)
                except Exception:
                    npv = np.nan
                if (
                    prev_npv is not None
                    and pd.notna(npv)
                    and pd.notna(prev_npv)
                    and npv * prev_npv < 0
                ):
                    found_bracket = True
                    try:
                        res = optimize.brentq(
                            calculate_npv,
                            a=prev_r,
                            b=r_scan,
                            args=(dates, solver_flows),
                            xtol=1e-6,
                            rtol=1e-6,
                            maxiter=100,
                        )
                        if np.isfinite(res) and res > -1.0:
                            irr_result = res
                            break
                    except Exception:
                        pass
                if pd.notna(npv):
                    prev_r, prev_npv = r_scan, npv

        # NPV never changes sign anywhere in (-1, 100000] -> no IRR exists for
        # this flow pattern (e.g. a net-loss series whose latest flows are
        # negative). That is a property of the flows, not a solver failure.
        if np.isnan(irr_result) and not found_bracket:
            logging.debug(
                "DEBUG IRR: No root exists for these flows (NPV never crosses zero); returning NaN."
            )
            return np.nan

    # 4. Final Validation and Return
    if not (
        isinstance(irr_result, (float, int))
        and np.isfinite(irr_result)
        and irr_result > -1.0
    ):
        # Format flows for readability
        readable_flows = [
            f"{d.strftime('%Y-%m-%d')}: {f:,.2f}" for d, f in zip(dates, solver_flows)
        ]
        # Truncate if too long
        if len(readable_flows) > 100:
            flow_str = f"[{', '.join(readable_flows[:10])} ... {', '.join(readable_flows[-10:])}] ({len(readable_flows)} flows)"
        else:
            flow_str = f"[{', '.join(readable_flows)}]"

        logging.warning(f"IRR Calculation Failed. Flows used: {flow_str}")
        return np.nan

    # --- DIAGNOSTIC LOGGING for extreme results ---
    if irr_result > 10.0:  # > 1000% p.a.
        logging.warning(
            f"Extreme IRR detected ({irr_result * 100:.1f}%). Investigating flows..."
        )
        # Log top 10 largest absolute flows for debugging
        top_flows = sorted(
            list(zip(dates, solver_flows)), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        readable_top = [f"{d.strftime('%Y-%m-%d')}: {f:,.2f}" for d, f in top_flows]
        logging.warning(f"Top Flows contributing to IRR: [{', '.join(readable_top)}]")

    return irr_result
