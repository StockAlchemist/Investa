# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          financial_ratios.py
 Purpose:       Calculate key financial ratios from fundamental data.

 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
from io import StringIO


def _get_statement_value(
    df: Optional[pd.DataFrame], item_name: str, period_column: str
) -> Optional[float]:
    """Safely retrieves a value from a financial statement DataFrame."""
    if (
        df is None
        or df.empty
        or item_name not in df.index
        or period_column not in df.columns
    ):
        return None
    val = df.loc[item_name, period_column]
    return float(val) if pd.notna(val) else None


def calculate_key_ratios_timeseries(
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    # cashflow_df: Optional[pd.DataFrame], # Not used in current set of ratios
    # ticker_info: Optional[Dict] = None, # Not used for historical series ratios
    # is_annual: bool = True # Not directly used, period determined by data
) -> pd.DataFrame:
    """
    Calculates a timeseries of key financial ratios.
    Assumes input DataFrames have periods as columns and financial items as index.
    """
    if (
        financials_df is None
        or financials_df.empty
        or balance_sheet_df is None
        or balance_sheet_df.empty
    ):
        logging.warning(
            "Financials or Balance Sheet data is missing/empty. Cannot calculate historical ratios."
        )
        return pd.DataFrame()

    try:
        # Convert columns to datetime objects, coercing errors, and drop NaT values
        fin_periods_dt = pd.to_datetime(financials_df.columns, errors="coerce").dropna()
        bs_periods_dt = pd.to_datetime(
            balance_sheet_df.columns, errors="coerce"
        ).dropna()

        if fin_periods_dt.empty or bs_periods_dt.empty:
            logging.warning(
                "No valid period columns found in financial statements for ratio calculation."
            )
            return pd.DataFrame()

        # Find common periods (as datetime objects)
        common_periods_dt = sorted(
            list(set(fin_periods_dt).intersection(set(bs_periods_dt)))
        )

    except Exception as e:
        logging.error(f"Could not parse period columns for ratio calculation: {e}")
        return pd.DataFrame()

    if not common_periods_dt:
        logging.warning(
            "No common periods found between income statement and balance sheet for ratio calculation."
        )
        return pd.DataFrame()

    ratios_data_list: List[Dict] = []  # Ensure type for list of dicts

    for i, period_dt in enumerate(common_periods_dt):
        # Find the original column name string that corresponds to this datetime period
        # This is necessary because yfinance might return columns like '2023-12-31' or 'TTM'
        # We need to match based on the parsed datetime.
        period_str_fin = next(
            (
                col
                for col in financials_df.columns
                if pd.to_datetime(col, errors="coerce") == period_dt
            ),
            None,
        )
        period_str_bs = next(
            (
                col
                for col in balance_sheet_df.columns
                if pd.to_datetime(col, errors="coerce") == period_dt
            ),
            None,
        )

        if not period_str_fin or not period_str_bs:
            logging.debug(
                f"Skipping period {period_dt.strftime('%Y-%m-%d')} due to missing original column name match."
            )
            continue

        current_ratios: Dict[str, Any] = {"Period": period_dt}  # Start with Period

        # Profitability
        revenue = _get_statement_value(financials_df, "Total Revenue", period_str_fin)
        cost_of_revenue = _get_statement_value(
            financials_df, "Cost Of Revenue", period_str_fin
        )
        gross_profit = _get_statement_value(
            financials_df, "Gross Profit", period_str_fin
        )
        if gross_profit is None and revenue is not None and cost_of_revenue is not None:
            gross_profit = revenue - cost_of_revenue
        net_income = _get_statement_value(financials_df, "Net Income", period_str_fin)
        if net_income is None:
            net_income = _get_statement_value(
                financials_df, "Net Income From Continuing Ops", period_str_fin
            )

        total_equity = _get_statement_value(
            balance_sheet_df, "Stockholders Equity", period_str_bs
        ) or _get_statement_value(
            balance_sheet_df, "Total Stockholder Equity", period_str_bs
        ) or _get_statement_value(
            balance_sheet_df, "Total Equity Gross Minority Interest", period_str_bs
        )
        total_assets = _get_statement_value(
            balance_sheet_df, "Total Assets", period_str_bs
        )

        avg_equity, avg_assets = total_equity, total_assets
        if i > 0:
            prev_period_dt = common_periods_dt[i - 1]
            prev_period_str_bs = next(
                (
                    col
                    for col in balance_sheet_df.columns
                    if pd.to_datetime(col, errors="coerce") == prev_period_dt
                ),
                None,
            )
            if prev_period_str_bs:
                prev_equity = _get_statement_value(
                    balance_sheet_df, "Total Stockholder Equity", prev_period_str_bs
                )
                prev_assets = _get_statement_value(
                    balance_sheet_df, "Total Assets", prev_period_str_bs
                )
                if total_equity is not None and prev_equity is not None:
                    avg_equity = (total_equity + prev_equity) / 2
                if total_assets is not None and prev_assets is not None:
                    avg_assets = (total_assets + prev_assets) / 2

        current_ratios["Gross Profit Margin (%)"] = (
            (gross_profit / revenue) * 100
            if revenue and revenue != 0 and gross_profit is not None
            else np.nan
        )
        current_ratios["Net Profit Margin (%)"] = (
            (net_income / revenue) * 100
            if revenue and revenue != 0 and net_income is not None
            else np.nan
        )
        current_ratios["Return on Equity (ROE) (%)"] = (
            (net_income / avg_equity) * 100
            if avg_equity and avg_equity != 0 and net_income is not None
            else np.nan
        )
        current_ratios["Return on Assets (ROA) (%)"] = (
            (net_income / avg_assets) * 100
            if avg_assets and avg_assets != 0 and net_income is not None
            else np.nan
        )

        # Liquidity
        current_assets = _get_statement_value(
            balance_sheet_df, "Current Assets", period_str_bs
        ) or _get_statement_value(
            balance_sheet_df, "Total Current Assets", period_str_bs
        )
        current_liabilities = _get_statement_value(
            balance_sheet_df, "Current Liabilities", period_str_bs
        ) or _get_statement_value(
            balance_sheet_df, "Total Current Liabilities", period_str_bs
        )
        inventory = _get_statement_value(balance_sheet_df, "Inventory", period_str_bs)
        current_ratios["Current Ratio"] = (
            current_assets / current_liabilities
            if current_liabilities
            and current_liabilities != 0
            and current_assets is not None
            else np.nan
        )
        current_ratios["Quick Ratio"] = (
            ((current_assets - (inventory or 0)) / current_liabilities)
            if current_liabilities
            and current_liabilities != 0
            and current_assets is not None
            else np.nan
        )

        # Solvency
        total_liab = _get_statement_value(
            balance_sheet_df, "Total Liabilities Net Minority Interest", period_str_bs
        ) or _get_statement_value(
            balance_sheet_df, "Total Liab", period_str_bs
        ) or _get_statement_value(balance_sheet_df, "Total Liabilities", period_str_bs)
        current_ratios["Debt-to-Equity Ratio"] = (
            total_liab / total_equity
            if total_equity and total_equity != 0 and total_liab is not None
            else np.nan
        )

        ebit = _get_statement_value(financials_df, "Ebit", period_str_fin)
        interest_exp = _get_statement_value(
            financials_df, "Interest Expense", period_str_fin
        )
        if interest_exp is not None and interest_exp < 0:
            interest_exp = abs(interest_exp)
        current_ratios["Interest Coverage Ratio"] = (
            ebit / interest_exp
            if interest_exp and interest_exp != 0 and ebit is not None
            else np.nan
        )

        # Efficiency
        current_ratios["Asset Turnover"] = (
            revenue / avg_assets
            if avg_assets and avg_assets != 0 and revenue is not None
            else np.nan
        )

        ratios_data_list.append(current_ratios)

    if not ratios_data_list:
        return pd.DataFrame()

    ratios_df = pd.DataFrame(ratios_data_list).set_index("Period")
    return ratios_df.sort_index(ascending=False)


def calculate_current_valuation_ratios(
    ticker_info: Optional[Dict],
    financials_df_latest_annual: Optional[pd.DataFrame] = None,
    balance_sheet_df_latest_annual: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[float]]:
    """Calculates point-in-time valuation ratios."""
    ratios: Dict[str, Optional[float]] = {
        "P/E Ratio (TTM)": np.nan,
        "Forward P/E Ratio": np.nan,
        "Price-to-Sales (P/S) Ratio (TTM)": np.nan,
        "Price-to-Book (P/B) Ratio (MRQ)": np.nan,
        "Dividend Yield (%)": np.nan,
        "Enterprise Value to EBITDA": np.nan,
    }
    if not ticker_info:
        return ratios

    current_price = ticker_info.get("currentPrice") or ticker_info.get(
        "regularMarketPrice"
    )
    market_cap = ticker_info.get("marketCap")

    ratios["P/E Ratio (TTM)"] = ticker_info.get("trailingPE")
    ratios["Forward P/E Ratio"] = ticker_info.get("forwardPE")
    if ticker_info.get("dividendYield") is not None:
        ratios["Dividend Yield (%)"] = ticker_info["dividendYield"]

    trailing_revenue = ticker_info.get(
        "totalRevenue"
    )  # yfinance info often has 'totalRevenue' for TTM
    if market_cap and trailing_revenue and trailing_revenue != 0:
        ratios["Price-to-Sales (P/S) Ratio (TTM)"] = market_cap / trailing_revenue

    book_value_per_share = ticker_info.get("bookValue")
    if current_price and book_value_per_share and book_value_per_share != 0:
        ratios["Price-to-Book (P/B) Ratio (MRQ)"] = current_price / book_value_per_share
    elif (
        market_cap
        and balance_sheet_df_latest_annual is not None
        and not balance_sheet_df_latest_annual.empty
    ):
        # Fallback to marketCap / latest total equity if bookValue per share is not available
        latest_bs_period = balance_sheet_df_latest_annual.columns[0]
        total_equity_latest = _get_statement_value(
            balance_sheet_df_latest_annual, "Total Stockholder Equity", latest_bs_period
        )
        if total_equity_latest and total_equity_latest != 0:
            ratios["Price-to-Book (P/B) Ratio (MRQ)"] = market_cap / total_equity_latest

    ratios["Enterprise Value to EBITDA"] = ticker_info.get("enterpriseToEbitda")
    return ratios


def calculate_wacc(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.045,  # Default to ~4.5% if not provided
    market_return: float = 0.09,    # Default to 9%
    default_tax_rate: float = 0.21
) -> Dict[str, Any]:
    """
    Calculates the Weighted Average Cost of Capital (WACC).
    """
    try:
        # 1. Cost of Equity (CAPM)
        beta = ticker_info.get("beta")
        if beta is None or pd.isna(beta):
            beta = 1.0  # Default to market beta
            logging.debug(f"Beta missing for {ticker_info.get('symbol')}, using 1.0")
            
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        
        # 2. Cost of Debt
        cost_of_debt = 0.05 # Default 5%
        tax_rate = default_tax_rate
        
        total_debt = ticker_info.get("totalDebt")
        interest_expense = None
        income_tax_expense = None
        pretax_income = None
        
        if financials_df is not None and not financials_df.empty:
            latest_period = financials_df.columns[0]
            interest_expense = _get_statement_value(financials_df, "Interest Expense", latest_period)
            income_tax_expense = _get_statement_value(financials_df, "Tax Provision", latest_period)
            pretax_income = _get_statement_value(financials_df, "Pretax Income", latest_period)
            
            if interest_expense and total_debt and total_debt > 0:
                cost_of_debt = abs(interest_expense) / total_debt
            
            if income_tax_expense and pretax_income and pretax_income > 0:
                tax_rate = income_tax_expense / pretax_income
        
        # 3. Weights
        market_cap = ticker_info.get("marketCap")
        if not market_cap:
            return {"wacc": max(0.075, cost_of_equity), "method": "Cost of Equity (No Market Cap)"}
            
        total_value = market_cap + (total_debt or 0)
        weight_equity = market_cap / total_value
        weight_debt = (total_debt or 0) / total_value
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        wacc = max(0.075, wacc)
        
        return {
            "wacc": wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "tax_rate": tax_rate,
            "weight_equity": weight_equity,
            "weight_debt": weight_debt,
            "beta": beta,
            "method": "WACC"
        }
    except Exception as e:
        logging.error(f"Error calculating WACC: {e}")
        return {"wacc": 0.10, "method": "Default (10%) due to error"}


def estimate_growth_rate(
    financials_df: Optional[pd.DataFrame],
    ticker_info: Optional[Dict[str, Any]] = None,
    item_name: str = "Net Income",
    years: int = 5
) -> float:
    """Attempts to estimate a historical growth rate for a financial item."""
    values = []
    
    # 1. Try to get ANALYST EXPECTED GROWTH (Priority)
    if ticker_info:
        # Check for our injected analyst data
        # We average '0y' (Current Year) and '+1y' (Next Year) estimates
        analyst_ee = ticker_info.get("_earnings_estimate", {})
        if analyst_ee:
            expected_rates = []
            for p in ["0y", "+1y"]:
                row = analyst_ee.get(p)
                if row and "growth" in row and row["growth"] is not None:
                    expected_rates.append(float(row["growth"]))
            
            if expected_rates:
                avg_expected = sum(expected_rates) / len(expected_rates)
                return avg_expected
        
        # Fallback to standard info fields if specific estimates missing
        g = ticker_info.get("earningsGrowth") or ticker_info.get("revenueGrowth")
        if g is not None:
            return float(g)

    # 2. Try to calculate historical CAGR if analyst data is not available
    # ... (keeping existing historical CAGR logic as fallback)
    if financials_df is not None and not financials_df.empty:
        try:
            # Sort columns to be chronological
            cols = sorted(financials_df.columns, key=lambda x: pd.to_datetime(x, errors="coerce"))
            recent_dated = []
            for col in cols:
                val = _get_statement_value(financials_df, item_name, col)
                if val is not None and val > 0:
                    recent_dated.append((pd.to_datetime(col), val))
            
            if len(recent_dated) >= 2:
                # Target window: last 3 years
                end_date, end_val = recent_dated[-1]
                
                # Find the starting point ~3 years before the end point
                start_idx = 0
                for i in range(len(recent_dated) - 2, -1, -1):
                    d, v = recent_dated[i]
                    years_diff = (end_date - d).days / 365.25
                    if years_diff >= 2.5: # approx 3 years
                        start_idx = i
                        break
                
                start_date, start_val = recent_dated[start_idx]
                
                # Calculate CAGR
                n_years = (end_date - start_date).days / 365.25
                if n_years > 0.5: 
                    return (end_val / start_val) ** (1 / n_years) - 1
        except Exception:
            pass

    # 3. Final Default
    return 0.05

def estimate_fcf_margin(
    financials_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    years: int = 5
) -> float:
    """
    Estimates a normalized Free Cash Flow margin based on historical data.
    """
    if (
        financials_df is None
        or financials_df.empty
        or cashflow_df is None
        or cashflow_df.empty
    ):
        return 0.05  # Default conservative 5%

    try:
        # Find common columns/periods
        common_cols = sorted(
            list(set(financials_df.columns).intersection(set(cashflow_df.columns))),
            key=lambda x: pd.to_datetime(x, errors="coerce")
        )
        
        margins = []
        for col in common_cols[-years:]: # Look at last N years
            rev = _get_statement_value(financials_df, "Total Revenue", col)
            ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", col)
            capex = _get_statement_value(cashflow_df, "Capital Expenditure", col)
            
            if rev and rev > 0 and ocf is not None and capex is not None:
                # Capex is usually negative
                fcf = ocf + capex 
                margin = fcf / rev
                if margin > 0:
                    margins.append(margin)
        
        if margins:
            return sum(margins) / len(margins)
            
    except Exception as e:
        logging.warning(f"Failed to estimate FCF margin: {e}")
        
    return 0.05  # Fallback

def run_monte_carlo_dcf(
    ticker_info: Dict[str, Any],
    base_fcf: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for DCF."""
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares:
            return {}

        # 1. Generate stochastic variables
        # Growth Rate: Normal distribution (20% relative std dev)
        growth_samples = np.random.normal(base_growth, abs(base_growth) * 0.2, iterations)
        # Discount Rate: Normal distribution (10% relative std dev)
        discount_samples = np.random.normal(base_discount, abs(base_discount) * 0.1, iterations)
        
        # Ensure rates are sensible (floor at 7.5% for discount, 0% for growth)
        growth_samples = np.maximum(0.0, growth_samples)
        discount_samples = np.maximum(0.075, discount_samples)
        
        # Apply 40% cap to growth samples for stability
        growth_samples = np.minimum(0.40, growth_samples)

        # 2. Vectorized Projections
        # Shape: (iterations, projection_years)
        years = np.arange(1, projection_years + 1)
        
        # Linear Fade: Growth trends from growth_sample to terminal_growth over projection period
        fade_factors = (years - 1) / (projection_years - 1) if projection_years > 1 else np.array([0])
        # yearly_growths shape: (iterations, projection_years)
        yearly_growths = growth_samples[:, None] - (growth_samples[:, None] - terminal_growth) * fade_factors
        
        # Calculate FCF for each year: base_fcf * cumprod(1 + g_i)
        fcf_projections = base_fcf * np.cumprod(1 + yearly_growths, axis=1)
        
        # 3. Present Value of FCFs
        # PV = FCF / (1 + r)^n
        pv_projections = fcf_projections / (1 + discount_samples[:, None]) ** years
        sum_pv_fcf = np.sum(pv_projections, axis=1)

        # 4. Terminal Value
        # TV = (FCF_last * (1 + g_term)) / (r - g_term)
        last_fcf = fcf_projections[:, -1]
        terminal_values = (last_fcf * (1 + terminal_growth)) / (discount_samples - terminal_growth)
        # PV of TV
        pv_terminal_values = terminal_values / (1 + discount_samples) ** projection_years

        # 5. Equity Value to Intrinsic Value
        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0
        enterprise_values = sum_pv_fcf + pv_terminal_values
        equity_values = enterprise_values + cash - debt
        intrinsic_values = equity_values / shares

        # 6. Generate Histogram for Probability Plot
        counts, edges = np.histogram(intrinsic_values, bins=40)
        midpoints = (edges[:-1] + edges[1:]) / 2
        
        # Apply Gaussian smoothing to make it look like a "bell curve"
        # 7-point Gaussian kernel for better smoothness
        kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        smoothed_counts = np.convolve(counts, kernel, mode='same')

        histogram = [
            {"price": float(p), "count": float(c)}
            for p, c in zip(midpoints, smoothed_counts)
        ]

        # 7. Extract Percentiles
        return {
            "bear": float(np.percentile(intrinsic_values, 10)),
            "base": float(np.percentile(intrinsic_values, 50)),
            "bull": float(np.percentile(intrinsic_values, 90)),
            "std_dev": float(np.std(intrinsic_values)),
            "histogram": histogram
        }
    except Exception as e:
        logging.error(f"Monte Carlo DCF failed: {e}")
        return {}


def run_monte_carlo_graham(
    eps: float,
    base_growth: float,
    base_bond_yield: float,
    iterations: int = 10000
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Graham's Formula."""
    try:
        # 1. Stochastic Variables
        # Growth Rate: 20% relative std dev
        growth_samples = np.random.normal(base_growth, abs(base_growth) * 0.2, iterations)
        # Bond Yield: 10% relative std dev
        yield_samples = np.random.normal(base_bond_yield, abs(base_bond_yield) * 0.1, iterations)
        
        # Floor for stability
        growth_samples = np.maximum(0.0, growth_samples)
        yield_samples = np.maximum(0.5, yield_samples)

        # 2. Vectorized Formula: V = (EPS * (8.5 + 2g) * 4.4) / Y
        # Note: base_growth for graham is usually passed as percentage (e.g. 5.0 for 5%)
        intrinsic_values = (eps * (8.5 + 2 * growth_samples) * 4.4) / yield_samples

        # 3. Generate Histogram
        counts, edges = np.histogram(intrinsic_values, bins=40)
        midpoints = (edges[:-1] + edges[1:]) / 2
        
        # Apply smoothing
        kernel = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        smoothed_counts = np.convolve(counts, kernel, mode='same')

        histogram = [
            {"price": float(p), "count": float(c)}
            for p, c in zip(midpoints, smoothed_counts)
        ]

        return {
            "bear": float(np.percentile(intrinsic_values, 10)),
            "base": float(np.percentile(intrinsic_values, 50)),
            "bull": float(np.percentile(intrinsic_values, 90)),
            "std_dev": float(np.std(intrinsic_values)),
            "histogram": histogram
        }
    except Exception as e:
        logging.error(f"Monte Carlo Graham failed: {e}")
        return {}


def calculate_intrinsic_value_dcf(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    discount_rate: Optional[float] = None,
    growth_rate: Optional[float] = None,
    projection_years: int = 10,
    terminal_growth_rate: float = 0.02,
    target_fcf_margin: Optional[float] = None,
    fcf: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs a Discounted Cash Flow (DCF) valuation.
    """
    try:
        current_revenue = ticker_info.get("totalRevenue")
        model_method = "DCF"
        
        # 1. Base FCF
        if fcf is None:
            fcf = ticker_info.get("freeCashflow")
            if fcf is None and cashflow_df is not None and not cashflow_df.empty:
                latest_cf_period = cashflow_df.columns[0]
                ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", latest_cf_period)
                capex = _get_statement_value(cashflow_df, "Capital Expenditure", latest_cf_period)
                if ocf is not None and capex is not None:
                    fcf = ocf + capex # Capex is usually negative in YF
        
        # Fallback for Negative FCF: Revenue-based estimation
        used_fcf_margin = None
        if (fcf is None or fcf <= 0) and current_revenue and current_revenue > 0:
            if target_fcf_margin is None:
                target_fcf_margin = estimate_fcf_margin(financials_df, cashflow_df)
            
            fcf = current_revenue * target_fcf_margin
            model_method = "Revenue-based DCF"
            used_fcf_margin = target_fcf_margin

        if fcf is None or fcf <= 0:
            return {"error": "Negative or missing Free Cash Flow"}
            
        # 2. Discount Rate (WACC)
        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = wacc_res["wacc"]
        else:
            # Apply stability floor even to provided discount rate
            discount_rate = max(0.075, discount_rate)
            
        # 3. Growth Rate
        if growth_rate is None:
            growth_rate = estimate_growth_rate(financials_df, ticker_info=ticker_info, item_name="Net Income")
            
        # We cap the input growth at 40% for multi-year CAGR stability.
        # This prevents astronomical valuations that assume physical impossibilities.
        applied_growth = min(growth_rate, 0.40)
        
        projected_fcf = []
        pv_fcf = []
        current_fcf = fcf
        
        for y in range(1, projection_years + 1):
            # Linear Fade: Growth trends from applied_growth to terminal_rate over projection period
            fade_factor = (y - 1) / (projection_years - 1) if projection_years > 1 else 0
            yearly_growth = applied_growth - (applied_growth - terminal_growth_rate) * fade_factor
            
            next_fcf = current_fcf * (1 + yearly_growth)
            projected_fcf.append(next_fcf)
            pv_fcf.append(next_fcf / ((1 + discount_rate) ** y))
            current_fcf = next_fcf
            
        # 5. Terminal Value
        # Ensure denominator is positive
        safe_discount = max(discount_rate, terminal_growth_rate + 0.01)
        terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (safe_discount - terminal_growth_rate)
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # 6. Enterprise Value to Equity Value
        enterprise_value = sum(pv_fcf) + pv_terminal_value
        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0
        equity_value = enterprise_value + cash - debt
        
        shares_outstanding = ticker_info.get("sharesOutstanding")
        if not shares_outstanding:
            return {"error": "Missing shares outstanding"}
            
        intrinsic_value = equity_value / shares_outstanding
        
        res = {
            "intrinsic_value": intrinsic_value,
            "model": model_method,
            "parameters": {
                "discount_rate": discount_rate,
                "growth_rate": growth_rate,
                "applied_growth": applied_growth,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": projection_years,
                "base_fcf": fcf,
                "fcf_margin": used_fcf_margin
            }
        }
        if growth_rate > 0.40:
            res["parameters"]["note"] = f"Growth capped at 40% for DCF stability; linear fade over {projection_years}y applied"
        else:
            res["parameters"]["note"] = f"Linear growth fade over {projection_years}y applied towards terminal rate"
        return res
    except Exception as e:
        return {"error": f"DCF calculation failed: {str(e)}"}


def calculate_intrinsic_value_graham(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame] = None,
    growth_rate: Optional[float] = None,
    eps: Optional[float] = None,
    bond_yield: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculates intrinsic value using Benjamin Graham's Revised Formula:
    V = (EPS * (8.5 + 2g) * 4.4) / Y
    Where:
    - EPS: Trailing 12 months Earnings Per Share
    - 8.5: P/E base for a no-growth company
    - g: Reasonably expected 7 to 10 year growth rate
    - 4.4: Average yield of high-grade corporate bonds in 1962
    - Y: Current yield on AAA corporate bonds (using 10Y Treasury yield as proxy)
    """
    try:
        if growth_rate is None:
            # Estimate growth pct for Graham (expecting percentage e.g. 5.0 for 5%)
            growth_rate = estimate_growth_rate(financials_df, ticker_info=ticker_info, item_name="Net Income") * 100 
            
        # Risk-free rate (10Y Treasury) as proxy for Y
        if bond_yield is None:
            bond_yield = 4.5 # Default 4.5%
        
        if eps is None:
            eps = ticker_info.get("trailingEps")
            
        # Fallback for Negative EPS: Book Value
        if (eps is None or eps <= 0) and balance_sheet_df is not None and not balance_sheet_df.empty:
            latest_bs_period = balance_sheet_df.columns[0]
            total_equity = _get_statement_value(balance_sheet_df, "Total Stockholder Equity", latest_bs_period) or \
                           _get_statement_value(balance_sheet_df, "Total Equity Gross Minority Interest", latest_bs_period)
            shares = ticker_info.get("sharesOutstanding")
            
            if total_equity and shares and shares > 0:
                book_value = total_equity / shares
                if book_value > 0:
                    return {
                        "intrinsic_value": book_value,
                        "model": "Book Value",
                        "parameters": {
                            "book_value_per_share": book_value,
                            "note": "Used because EPS is negative/missing"
                        }
                    }

        if eps is None or eps <= 0:
            return {"error": "Negative or missing EPS"}
        
        # Graham's formula is extremely sensitive to hyper-growth.
        # We cap the 'g' used in the FORMULA to 30% for mathematical logic,
        # otherwise the intrinsic value becomes infinite.
        applied_growth = min(growth_rate, 30.0)
        intrinsic_value = (eps * (8.5 + 2 * applied_growth) * 4.4) / bond_yield
        
        res = {
            "intrinsic_value": intrinsic_value,
            "model": "Graham's Revised Formula",
            "parameters": {
                "eps": eps,
                "growth_rate_pct": growth_rate,
                "applied_growth_pct": applied_growth,
                "bond_yield_proxy": bond_yield
            }
        }
        if growth_rate > 30.0:
            res["parameters"]["note"] = "Growth capped at 30% for Graham formula sanity"
            
        return res
    except Exception as e:
        return {"error": f"Graham calculation failed: {str(e)}"}


def get_comprehensive_intrinsic_value(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Consolidates multiple intrinsic value models into a single advice object.
    """
    overrides = overrides or {}
    
    # Extract DCF overrides
    dcf_discount = overrides.get("dcf_discount_rate")
    dcf_growth = overrides.get("dcf_growth_rate")
    dcf_terminal = overrides.get("dcf_terminal_growth", 0.02)
    dcf_projection = int(overrides.get("dcf_projection_years", 10))
    dcf_fcf = overrides.get("dcf_fcf")
    target_fcf_margin = overrides.get("target_fcf_margin")
    
    # Extract Graham overrides
    graham_growth = overrides.get("graham_growth_rate")
    graham_eps = overrides.get("graham_eps")
    graham_bond_yield = overrides.get("graham_bond_yield")

    dcf_res = calculate_intrinsic_value_dcf(
        ticker_info, financials_df, balance_sheet_df, cashflow_df,
        discount_rate=dcf_discount,
        growth_rate=dcf_growth,
        projection_years=dcf_projection,
        terminal_growth_rate=dcf_terminal,
        fcf=dcf_fcf,
        target_fcf_margin=target_fcf_margin
    )
    
    graham_res = calculate_intrinsic_value_graham(
        ticker_info, financials_df, balance_sheet_df,
        growth_rate=graham_growth,
        eps=graham_eps,
        bond_yield=graham_bond_yield
    )
    
    current_price = ticker_info.get("currentPrice") or ticker_info.get("regularMarketPrice")
    
    # --- Monte Carlo Simulations ---
    dcf_mc = {}
    if "intrinsic_value" in dcf_res:
        params = dcf_res["parameters"]
        # Use applied growth for MC if available
        mc_growth = params.get("applied_growth", params["growth_rate"])
        dcf_mc = run_monte_carlo_dcf(
            ticker_info,
            params["base_fcf"],
            mc_growth,
            params["discount_rate"],
            params["projection_years"],
            params["terminal_growth_rate"]
        )
        dcf_res["mc"] = dcf_mc

    graham_mc = {}
    if "intrinsic_value" in graham_res:
        params = graham_res["parameters"]
        if "eps" in params:
            # Use applied growth for MC if available
            mc_growth_pct = params.get("applied_growth_pct", params.get("growth_rate_pct", 0))
            graham_mc = run_monte_carlo_graham(
                params["eps"],
                mc_growth_pct,
                params["bond_yield_proxy"]
            )
            graham_res["mc"] = graham_mc

    results = {
        "current_price": current_price,
        "models": {
            "dcf": dcf_res,
            "graham": graham_res
        }
    }
    
    # Calculate weighted average and range
    valid_values = []
    bear_values = []
    bull_values = []
    
    if "intrinsic_value" in dcf_res:
        valid_values.append(dcf_res["intrinsic_value"])
        if dcf_mc:
            bear_values.append(dcf_mc["bear"])
            bull_values.append(dcf_mc["bull"])
            
    if "intrinsic_value" in graham_res:
        valid_values.append(graham_res["intrinsic_value"])
        if graham_mc:
            bear_values.append(graham_mc["bear"])
            bull_values.append(graham_mc["bull"])
        
    if valid_values:
        avg_intrinsic = sum(valid_values) / len(valid_values)
        results["average_intrinsic_value"] = avg_intrinsic
        
        # Probabilistic range
        if bear_values and bull_values:
            results["range"] = {
                "bear": sum(bear_values) / len(bear_values),
                "bull": sum(bull_values) / len(bull_values)
            }
            
        if current_price:
            results["margin_of_safety_pct"] = ((avg_intrinsic - current_price) / current_price) * 100
            
    return results
