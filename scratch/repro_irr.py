import numpy as np
from datetime import date, datetime
import logging
import sys

# Mocking the calculate_irr and calculate_npv from finutils
from scipy import optimize
import warnings

def calculate_npv(rate, dates, cash_flows):
    if not np.isfinite(rate): return np.nan
    base = 1.0 + rate
    if base <= 1e-9: return np.nan
    start_date = dates[0]
    npv = 0.0
    for i in range(len(cash_flows)):
        time_delta_years = (dates[i] - start_date).days / 365.0
        denominator = base**time_delta_years
        npv += cash_flows[i] / denominator
    return npv

def calculate_irr(dates, cash_flows):
    readable_flows = [f"{d.strftime('%Y-%m-%d')}: {f:,.2f}" for d, f in zip(dates, cash_flows)]
    flow_str = f"[{', '.join(readable_flows)}]"
    
    try:
        irr_result = optimize.newton(
            calculate_npv, x0=0.1, args=(dates, cash_flows), tol=1e-6, maxiter=100
        )
    except:
        try:
            lower_bound, upper_bound = -0.9999, 10000.0
            irr_result = optimize.brentq(
                calculate_npv,
                a=lower_bound,
                b=upper_bound,
                args=(dates, cash_flows),
                xtol=1e-6,
                maxiter=100,
            )
        except:
            print(f"WARNING:root:IRR Calculation Failed. Flows used: {flow_str}")
            return np.nan
    return irr_result

# Data from user
dates = [
    date(2025, 11, 19), date(2025, 11, 20), date(2025, 12, 1),
    date(2025, 12, 4), date(2025, 12, 24), date(2026, 1, 1),
    date(2026, 2, 2), date(2026, 2, 9), date(2026, 2, 10),
    date(2026, 3, 1), date(2026, 3, 26), date(2026, 3, 27),
    date(2026, 4, 1)
]
flows = [
    -18472.90, -0.00, -8.90, 0.00, -0.00, -2.41,
    -0.00, 6.51, -6.51, -0.01, 2.24, -2.24, -0.01
]

print("Testing with current flows (no terminal value):")
res = calculate_irr(dates, flows)
print(f"Result: {res}")

# Now add a terminal value
print("\nTesting with terminal value (e.g. 19000.0) on 2026-04-22:")
dates_with_terminal = dates + [date(2026, 4, 22)]
flows_with_terminal = flows + [19000.0]
res_terminal = calculate_irr(dates_with_terminal, flows_with_terminal)
print(f"Result: {res_terminal * 100:.2f}%")
