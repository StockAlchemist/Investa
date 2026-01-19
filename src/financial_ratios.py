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
            return {"wacc": cost_of_equity, "method": "Cost of Equity (No Market Cap)"}
            
        total_value = market_cap + (total_debt or 0)
        weight_equity = market_cap / total_value
        weight_debt = (total_debt or 0) / total_value
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        
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
    item_name: str = "Net Income",
    years: int = 5
) -> float:
    """Attempts to estimate a historical growth rate for a financial item."""
    if financials_df is None or financials_df.empty:
        return 0.05 # Default 5%
        
    try:
        # Sort columns to be chronological
        cols = sorted(financials_df.columns, key=lambda x: pd.to_datetime(x, errors="coerce"))
        values = []
        for col in cols:
            val = _get_statement_value(financials_df, item_name, col)
            if val is not None and val > 0:
                values.append(val)
        
        if len(values) < 2:
            return 0.05
            
        # Simplistic CAGR
        start_val = values[0]
        end_val = values[-1]
        n_periods = len(values) - 1
        
        cagr = (end_val / start_val) ** (1/n_periods) - 1
        # Remove 15% cap at user request. Keeping floor at 0% for stability.
        return max(0.0, cagr)
    except Exception:
        return 0.05


def calculate_intrinsic_value_dcf(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    discount_rate: Optional[float] = None,
    growth_rate: Optional[float] = None,
    projection_years: int = 5,
    terminal_growth_rate: float = 0.02,
    fcf: Optional[float] = None
) -> Dict[str, Any]:
    """
    Performs a Discounted Cash Flow (DCF) valuation.
    """
    try:
        # 1. Base FCF
        if fcf is None:
            fcf = ticker_info.get("freeCashflow")
            if fcf is None and cashflow_df is not None and not cashflow_df.empty:
                latest_cf_period = cashflow_df.columns[0]
                ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", latest_cf_period)
                capex = _get_statement_value(cashflow_df, "Capital Expenditure", latest_cf_period)
                if ocf is not None and capex is not None:
                    fcf = ocf + capex # Capex is usually negative in YF
        
        if fcf is None or fcf <= 0:
            return {"error": "Negative or missing Free Cash Flow"}
            
        # 2. Discount Rate (WACC)
        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = wacc_res["wacc"]
            
        # 3. Growth Rate
        if growth_rate is None:
            growth_rate = estimate_growth_rate(financials_df, "Net Income")
            
        # 4. Projections
        projected_fcf = []
        pv_fcf = []
        current_fcf = fcf
        for y in range(1, projection_years + 1):
            next_fcf = current_fcf * (1 + growth_rate)
            projected_fcf.append(next_fcf)
            pv_fcf.append(next_fcf / ((1 + discount_rate) ** y))
            current_fcf = next_fcf
            
        # 5. Terminal Value
        terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
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
        
        return {
            "intrinsic_value": intrinsic_value,
            "model": "DCF",
            "parameters": {
                "discount_rate": discount_rate,
                "growth_rate": growth_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": projection_years,
                "base_fcf": fcf
            }
        }
    except Exception as e:
        return {"error": f"DCF calculation failed: {str(e)}"}


def calculate_intrinsic_value_graham(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
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
            growth_rate = estimate_growth_rate(financials_df, "Net Income") * 100 # Convert to percentage
            
        # Risk-free rate (10Y Treasury) as proxy for Y
        if bond_yield is None:
            bond_yield = 4.5 # Default 4.5%
        
        if eps is None:
            eps = ticker_info.get("trailingEps")
            
        if eps is None or eps <= 0:
            return {"error": "Negative or missing EPS"}
        
        intrinsic_value = (eps * (8.5 + 2 * growth_rate) * 4.4) / bond_yield
        
        return {
            "intrinsic_value": intrinsic_value,
            "model": "Graham's Revised Formula",
            "parameters": {
                "eps": eps,
                "growth_rate_pct": growth_rate,
                "bond_yield_proxy": bond_yield
            }
        }
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
    dcf_projection = int(overrides.get("dcf_projection_years", 5))
    dcf_fcf = overrides.get("dcf_fcf")
    
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
        fcf=dcf_fcf
    )
    
    graham_res = calculate_intrinsic_value_graham(
        ticker_info, financials_df,
        growth_rate=graham_growth,
        eps=graham_eps,
        bond_yield=graham_bond_yield
    )
    
    current_price = ticker_info.get("currentPrice") or ticker_info.get("regularMarketPrice")
    
    results = {
        "current_price": current_price,
        "models": {
            "dcf": dcf_res,
            "graham": graham_res
        }
    }
    
    # Calculate a simple average if both succeeded
    valid_values = []
    if "intrinsic_value" in dcf_res:
        valid_values.append(dcf_res["intrinsic_value"])
    if "intrinsic_value" in graham_res:
        valid_values.append(graham_res["intrinsic_value"])
        
    if valid_values:
        avg_intrinsic = sum(valid_values) / len(valid_values)
        results["average_intrinsic_value"] = avg_intrinsic
        if current_price:
            results["margin_of_safety_pct"] = (1 - (current_price / avg_intrinsic)) * 100
            
    return results
