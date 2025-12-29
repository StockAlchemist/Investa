# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          financial_ratios.py
 Purpose:       Calculate key financial ratios from fundamental data.

 Author:        Gemini Code Assist

 Created:       [Current Date]
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any


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
            balance_sheet_df, "Total Stockholder Equity", period_str_bs
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
            balance_sheet_df, "Total Current Assets", period_str_bs
        )
        current_liabilities = _get_statement_value(
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
