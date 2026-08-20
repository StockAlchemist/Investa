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
from typing import Dict, Optional, List, Any, Tuple

# ---------------------------------------------------------------------------
# Valuation policy
# ---------------------------------------------------------------------------
# One place for every judgement call the models make, so the assumptions are
# auditable instead of buried as literals at their point of use. Measured with
# `scripts/intrinsic_value_lab.py` over the local fundamentals cache.

# Growth. A DCF is mostly a bet on this number, so it is the number most worth
# disciplining. Near-term analyst estimates cover 1-2 years and are extrapolated
# here over a decade, so they are shrunk hard toward a base rate rather than
# trusted at face value. The base rate is roughly long-run nominal GDP: the
# rate at which a mature business grows once its niche is saturated.
BASE_RATE_GROWTH = 0.04
# Weight on the firm's own measured growth vs. the base rate. Empirical growth
# persistence over 5-10y horizons is weak, hence a majority weight on the base.
GROWTH_SHRINK_WEIGHT = 0.35
# Hard band on the growth actually fed to a 10-year projection. 25% sustained
# for a decade is already a 9x business; the old 40% cap implied 29x.
MAX_PROJECTED_GROWTH = 0.25
MIN_PROJECTED_GROWTH = -0.05
# Years of history the growth measurements read. Five was yfinance's limit, not
# a choice: it made a decade-long projection out of a window too short to hold a
# downturn, so one good stretch set the trend. Ten spans a cycle, and the shrink
# and the band above are what keep a long high-growth run from being
# extrapolated forward regardless.
GROWTH_HISTORY_YEARS = 10
TERMINAL_GROWTH = 0.02

# Discount rate.
RISK_FREE_RATE = 0.045
EQUITY_MARKET_RETURN = 0.09
MIN_DISCOUNT_RATE = 0.08
MAX_DISCOUNT_RATE = 0.20
MAX_COST_OF_DEBT = 0.25  # above this the "interest / debt" ratio is a data error
# Small caps carry real financing and liquidity risk that beta does not capture.
SIZE_PREMIUM_BANDS: List[Tuple[float, float]] = [
    (3e8, 0.045),  # < $300M
    (2e9, 0.025),  # < $2B
    (1e10, 0.010),  # < $10B
]

# Eligibility. A valuation is refused rather than fabricated when the inputs
# cannot support one — coverage is not a goal in itself, and a fabricated number
# next to a real price is worse than a blank.
MIN_PRICE_FOR_VALUATION = 1.0
MIN_MARKET_CAP_FOR_VALUATION = 5e7
# Output sanity band, as a multiple of price. Values outside it are reported as
# clamped rather than silently dropped, so the caller can see the model failed.
MAX_IV_TO_PRICE = 5.0
MIN_IV_TO_PRICE = 0.1

# Blend weights for the central fair-value estimate.
#
# EPV is deliberately absent. It is a *floor* — the value of current earning
# power with no growth — and measured on the local cache it lands below the DCF
# for 75% of companies (median 0.38x price vs 0.64x). Averaging a floor into a
# central estimate does not make the estimate conservative, it makes it wrong:
# including it at 35% weight pushed the median margin of safety to -20%, i.e.
# the claim that the typical profitable company is a fifth overvalued. EPV is
# computed and reported on its own, where a floor is what the reader wants.
MODEL_BLEND_WEIGHTS = {"dcf": 0.60, "graham": 0.40}


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


# A share count is a level, not a flow: four quarters of "Diluted Average
# Shares" summed would report four times the company's shares. These rows are
# averaged over the window instead of added.
_LEVEL_ROW_MARKERS = ("Shares", "Share Issued")


def _is_level_row(label: Any) -> bool:
    text = str(label)
    return any(marker in text for marker in _LEVEL_ROW_MARKERS)


# Bounds on the *average* quarter in the window, which is what its span divided
# by its steps comes to. 12 weeks is the shortest quarter any calendar uses; the
# average never reaches the 16 weeks a 12/12/12/16 filer closes its year with,
# because the three siblings pulling it back down are in the same window. Across
# nine filers and 600 windows the real spread is 252 to 287 days for four
# quarters, so 14 weeks is a ceiling with room rather than a guess.
_MIN_QUARTER_SPAN_DAYS = 80
_MAX_QUARTER_SPAN_DAYS = 98


def _is_consecutive_window(newest: Any, oldest: Any, quarters: int) -> bool:
    """
    Whether the ends of a window this wide can be `quarters` consecutive quarters.

    Adjacency has to be checked, not assumed. The quarterly series is built by
    differencing a year-to-date ladder and *refuses* to difference across a
    missing rung, so a hole in it is the designed-for outcome rather than a
    surprise — and four columns either side of a hole are not a year. NVIDIA
    tagged one quarter twice, a day apart, and the four columns ending
    2011-01-30 spanned 183 days and reported $3.353bn of trailing revenue
    against the $3.543bn it filed.

    Measured between the outermost period *ends*, which is one quarter fewer
    than the window holds.
    """
    if pd.isna(newest) or pd.isna(oldest) or quarters < 2:
        return False
    steps = quarters - 1
    days = (newest - oldest).days
    return _MIN_QUARTER_SPAN_DAYS * steps <= days <= _MAX_QUARTER_SPAN_DAYS * steps


def to_trailing_twelve_months(
    statement_df: Optional[pd.DataFrame], min_quarters: int = 4
) -> Optional[pd.DataFrame]:
    """
    Turn a quarterly flow statement into trailing-twelve-month columns.

    A ratio that divides a flow by a level — return on equity, asset turnover —
    is meaningless on a raw quarter: one quarter of profit over the whole
    equity balance reads as a quarter of the real return, and would sit four
    times below the annual series for the same company. Summing the four
    quarters ending at each period restores the comparison, and keeps the
    quarterly *sampling* that makes a turn in the trend visible three quarters
    before the annual figure shows it.

    Columns are newest-first, as every statement in this system is. A period
    without four quarters behind it is dropped rather than annualised from
    fewer — a company's first three quarters are not a year — and so is one
    whose four columns are not four *consecutive* quarters, which is the same
    rule the quarterly series itself follows when it refuses to difference
    across a missing rung.
    """
    if statement_df is None or statement_df.empty:
        return statement_df

    periods = list(statement_df.columns)
    if len(periods) < min_quarters:
        return statement_df.iloc[:, :0]

    stamps = pd.to_datetime(pd.Index(periods), errors="coerce")

    columns: Dict[Any, pd.Series] = {}
    for i in range(len(periods) - min_quarters + 1):
        oldest = i + min_quarters - 1
        if not _is_consecutive_window(stamps[i], stamps[oldest], min_quarters):
            continue
        window = statement_df.iloc[:, i : oldest + 1]
        # `min_count` keeps a row that no quarter reported as NaN rather than
        # letting it sum to a confident zero.
        summed = window.sum(axis=1, min_count=min_quarters)
        averaged = window.mean(axis=1)
        columns[periods[i]] = summed.where(~summed.index.map(_is_level_row), averaged)

    if not columns:
        return statement_df.iloc[:, :0]
    return pd.DataFrame(columns)


def _price_on_or_before(prices_data: Any, dt: Any) -> Optional[float]:
    """Finds the split-consistent price on or immediately before `dt`."""
    if prices_data is None or dt is None:
        return None
    try:
        series = None
        if isinstance(prices_data, pd.DataFrame):
            for col in ["price", "Price", "Adj Close", "Close", "adj_close", "close"]:
                if col in prices_data.columns:
                    series = prices_data[col]
                    break
            if series is None and not prices_data.empty:
                series = prices_data.iloc[:, 0]
        elif isinstance(prices_data, pd.Series):
            series = prices_data
        elif isinstance(prices_data, dict):
            series = pd.Series(prices_data)

        if series is None or series.empty:
            return None

        # Convert index to datetime if needed
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series[series.index.notnull()]

        # Normalize timezone on index to naive UTC for safe comparison
        if getattr(series.index, "tz", None) is not None:
            try:
                series.index = series.index.tz_convert("UTC").tz_localize(None)
            except Exception:
                series.index = series.index.tz_localize(None)

        dt_ts = pd.to_datetime(dt)
        if dt_ts is None or pd.isna(dt_ts):
            return None
        if getattr(dt_ts, "tzinfo", None) is not None:
            try:
                dt_ts = dt_ts.tz_convert("UTC").tz_localize(None)
            except Exception:
                dt_ts = dt_ts.tz_localize(None)

        series = series.sort_index()
        sub = series.loc[:dt_ts]
        if sub.empty:
            return None
        last_dt = sub.index[-1]
        if (dt_ts - last_dt).days > 60:
            return None
        val = float(sub.iloc[-1])
        return val if val > 0 and np.isfinite(val) else None
    except Exception as e:
        logging.debug(f"_price_on_or_before error: {e}")
        return None


def calculate_key_ratios_timeseries(
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame] = None,
    periods_per_year: int = 1,
    prices_df: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Calculates a timeseries of key financial ratios.
    Assumes input DataFrames have periods as columns and financial items as index.

    `cashflow_df` adds free-cash-flow margin, operating cashflow and dividend yield.
    `prices_df` adds point-in-time historical valuation multiples (P/E, P/S, P/B, EV/EBITDA, EV/Sales, P/FCF).
    `periods_per_year` says how many columns make a year (4 for quarterly TTM, 1 for annual).
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
        fin_periods_dt = pd.to_datetime(financials_df.columns, errors="coerce").dropna()
        bs_periods_dt = pd.to_datetime(
            balance_sheet_df.columns, errors="coerce"
        ).dropna()

        if fin_periods_dt.empty or bs_periods_dt.empty:
            logging.warning(
                "No valid period columns found in financial statements for ratio calculation."
            )
            return pd.DataFrame()

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

    ratios_data_list: List[Dict] = []

    for i, period_dt in enumerate(common_periods_dt):
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

        current_ratios: Dict[str, Any] = {"Period": period_dt}

        # 1. Profitability & Operations
        revenue = _get_statement_value(financials_df, "Total Revenue", period_str_fin)
        cost_of_revenue = _get_statement_value(
            financials_df, "Cost Of Revenue", period_str_fin
        )
        gross_profit = _get_statement_value(
            financials_df, "Gross Profit", period_str_fin
        )
        if gross_profit is None and revenue is not None and cost_of_revenue is not None:
            gross_profit = revenue - cost_of_revenue

        operating_income = _get_statement_value(
            financials_df, "Operating Income", period_str_fin
        ) or _get_statement_value(
            financials_df, "Operating Revenue Or Loss", period_str_fin
        )

        ebit = _get_statement_value(financials_df, "Ebit", period_str_fin)
        if operating_income is None and ebit is not None:
            operating_income = ebit

        net_income = _get_statement_value(financials_df, "Net Income", period_str_fin)
        if net_income is None:
            net_income = _get_statement_value(
                financials_df, "Net Income From Continuing Ops", period_str_fin
            )

        total_equity = (
            _get_statement_value(balance_sheet_df, "Stockholders Equity", period_str_bs)
            or _get_statement_value(
                balance_sheet_df, "Total Stockholder Equity", period_str_bs
            )
            or _get_statement_value(
                balance_sheet_df, "Total Equity Gross Minority Interest", period_str_bs
            )
        )
        total_assets = _get_statement_value(
            balance_sheet_df, "Total Assets", period_str_bs
        )

        avg_equity, avg_assets = total_equity, total_assets
        if i >= periods_per_year:
            prev_period_dt = common_periods_dt[i - periods_per_year]
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
        current_ratios["Operating Margin (%)"] = (
            (operating_income / revenue) * 100
            if revenue and revenue != 0 and operating_income is not None
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

        # 2. Liquidity
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

        # 3. Solvency
        total_liab = (
            _get_statement_value(
                balance_sheet_df,
                "Total Liabilities Net Minority Interest",
                period_str_bs,
            )
            or _get_statement_value(balance_sheet_df, "Total Liab", period_str_bs)
            or _get_statement_value(
                balance_sheet_df, "Total Liabilities", period_str_bs
            )
        )
        current_ratios["Debt-to-Equity Ratio"] = (
            total_liab / total_equity
            if total_equity and total_equity != 0 and total_liab is not None
            else np.nan
        )

        lt_debt = (
            _get_statement_value(balance_sheet_df, "Long Term Debt", period_str_bs)
            or _get_statement_value(
                balance_sheet_df,
                "Long Term Debt And Capital Lease Obligation",
                period_str_bs,
            )
            or _get_statement_value(
                balance_sheet_df, "Long Term Debt Non Current", period_str_bs
            )
        )
        current_ratios["Long-Term Debt to Equity"] = (
            lt_debt / total_equity
            if total_equity and total_equity != 0 and lt_debt is not None
            else np.nan
        )

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

        # 4. Efficiency
        current_ratios["Asset Turnover"] = (
            revenue / avg_assets
            if avg_assets and avg_assets != 0 and revenue is not None
            else np.nan
        )

        # 5. ROIC
        pretax = _get_statement_value(financials_df, "Pretax Income", period_str_fin)
        tax = _get_statement_value(financials_df, "Tax Provision", period_str_fin)
        total_debt = _get_statement_value(balance_sheet_df, "Total Debt", period_str_bs)
        cash = _get_statement_value(
            balance_sheet_df, "Cash And Cash Equivalents", period_str_bs
        )
        roic = np.nan
        if pretax is not None and total_equity:
            operating_earnings = pretax + (interest_exp or 0.0)
            tax_rate = (
                (tax / pretax) if pretax and pretax > 0 and tax is not None else None
            )
            if tax_rate is None or not (0.0 <= tax_rate <= 0.6):
                tax_rate = 0.21
            nopat = operating_earnings * (1.0 - tax_rate)
            invested = total_equity + (total_debt or 0.0) - (cash or 0.0)
            if invested and invested != 0:
                candidate = nopat / invested
                if -2.0 < candidate < 3.0:
                    roic = candidate * 100.0
        current_ratios["Return on Invested Capital (ROIC) (%)"] = roic

        # 6. Cash Flow & Shares
        current_ratios["Free Cash Flow Margin (%)"] = np.nan
        current_ratios["Diluted Shares Outstanding"] = np.nan
        ocf, capex, div_paid = None, None, None
        period_str_cf = None
        if cashflow_df is not None and not cashflow_df.empty:
            period_str_cf = next(
                (
                    col
                    for col in cashflow_df.columns
                    if pd.to_datetime(col, errors="coerce") == period_dt
                ),
                None,
            )
            if period_str_cf is not None:
                ocf = _get_statement_value(
                    cashflow_df, "Operating Cash Flow", period_str_cf
                )
                capex = _get_statement_value(
                    cashflow_df, "Capital Expenditure", period_str_cf
                )
                div_paid = _get_statement_value(
                    cashflow_df, "Common Stock Dividend Paid", period_str_cf
                ) or _get_statement_value(
                    cashflow_df, "Cash Dividends Paid", period_str_cf
                )
                if ocf is not None and capex is not None and revenue:
                    current_ratios["Free Cash Flow Margin (%)"] = (
                        (ocf + capex) / revenue
                    ) * 100.0

        shares = _get_statement_value(
            financials_df, "Diluted Average Shares", period_str_fin
        ) or _get_statement_value(
            balance_sheet_df, "Ordinary Shares Number", period_str_bs
        )
        if shares:
            current_ratios["Diluted Shares Outstanding"] = shares

        # 7. Level figures (Revenue, Net Income, EPS) & YoY Growth
        current_ratios["Total Revenue"] = revenue
        current_ratios["Operating Income"] = operating_income
        current_ratios["Net Income"] = net_income

        eps = _get_statement_value(financials_df, "Diluted EPS", period_str_fin)
        if eps is None and net_income is not None and shares and shares > 0:
            eps = net_income / shares
        current_ratios["Diluted EPS"] = eps

        # YoY Growth rates against period from 1 year prior
        current_ratios["Revenue Growth YoY (%)"] = np.nan
        current_ratios["EPS Growth YoY (%)"] = np.nan
        if i >= periods_per_year:
            prev_dt = common_periods_dt[i - periods_per_year]
            prev_fin_col = next(
                (
                    col
                    for col in financials_df.columns
                    if pd.to_datetime(col, errors="coerce") == prev_dt
                ),
                None,
            )
            if prev_fin_col:
                prev_rev = _get_statement_value(
                    financials_df, "Total Revenue", prev_fin_col
                )
                if prev_rev and prev_rev > 0 and revenue is not None:
                    current_ratios["Revenue Growth YoY (%)"] = (
                        (revenue / prev_rev) - 1
                    ) * 100.0

                prev_eps = _get_statement_value(
                    financials_df, "Diluted EPS", prev_fin_col
                )
                if prev_eps and prev_eps > 0 and eps is not None and eps > 0:
                    current_ratios["EPS Growth YoY (%)"] = (
                        (eps / prev_eps) - 1
                    ) * 100.0

        # 8. Historical Valuation Multiples over Time (using historical price)
        price = _price_on_or_before(prices_df, period_dt)
        current_ratios["Price"] = price
        current_ratios["P/E Ratio"] = np.nan
        current_ratios["P/S Ratio"] = np.nan
        current_ratios["P/B Ratio"] = np.nan
        current_ratios["P/FCF Ratio"] = np.nan
        current_ratios["EV/EBITDA"] = np.nan
        current_ratios["EV/Sales"] = np.nan
        current_ratios["Dividend Yield (%)"] = np.nan

        if price is not None and shares and shares > 0:
            mkt_cap = price * shares
            if eps and eps > 0:
                current_ratios["P/E Ratio"] = price / eps
            elif net_income and net_income > 0:
                current_ratios["P/E Ratio"] = mkt_cap / net_income

            if revenue and revenue > 0:
                current_ratios["P/S Ratio"] = mkt_cap / revenue

            if total_equity and total_equity > 0:
                current_ratios["P/B Ratio"] = mkt_cap / total_equity

            fcf = (ocf + capex) if (ocf is not None and capex is not None) else None
            if fcf and fcf > 0:
                current_ratios["P/FCF Ratio"] = mkt_cap / fcf

            ev = mkt_cap + (total_debt or 0.0) - (cash or 0.0)
            if revenue and revenue > 0 and ev > 0:
                current_ratios["EV/Sales"] = ev / revenue

            ebitda = (
                _get_statement_value(financials_df, "EBITDA", period_str_fin)
                or _get_statement_value(financials_df, "Ebitda", period_str_fin)
                or _get_statement_value(financials_df, "Normalized EBITDA", period_str_fin)
            )
            if ebitda is None:
                base_ebit = ebit if ebit is not None else operating_income
                if base_ebit is not None:
                    depr = (
                        _get_statement_value(financials_df, "Reconciled Depreciation", period_str_fin)
                        or _get_statement_value(financials_df, "Depreciation And Amortization", period_str_fin)
                        or (
                            _get_statement_value(cashflow_df, "Depreciation And Amortization", period_str_cf)
                            if cashflow_df is not None and period_str_cf is not None else None
                        )
                        or (
                            _get_statement_value(cashflow_df, "Depreciation Amortization Depletion", period_str_cf)
                            if cashflow_df is not None and period_str_cf is not None else None
                        )
                        or (
                            _get_statement_value(cashflow_df, "Depreciation", period_str_cf)
                            if cashflow_df is not None and period_str_cf is not None else None
                        )
                    )
                    if depr is not None:
                        ebitda = base_ebit + abs(depr)
                    else:
                        ebitda = base_ebit
            if ebitda and ebitda > 0 and ev > 0:
                current_ratios["EV/EBITDA"] = ev / ebitda

            if div_paid is not None and mkt_cap > 0:
                current_ratios["Dividend Yield (%)"] = (abs(div_paid) / mkt_cap) * 100.0

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
    risk_free_rate: float = RISK_FREE_RATE,
    market_return: float = EQUITY_MARKET_RETURN,
    default_tax_rate: float = 0.21,
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
        # Yahoo betas on thin tickers run to absurd values in both directions;
        # outside this band the number describes the data, not the risk.
        beta = float(np.clip(beta, 0.3, 3.0))

        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

        # Beta is estimated from price history, which says little about the
        # financing and liquidity risk of a small company.
        market_cap_for_size = ticker_info.get("marketCap")
        size_premium = 0.0
        if market_cap_for_size:
            for threshold, premium in SIZE_PREMIUM_BANDS:
                if market_cap_for_size < threshold:
                    size_premium = premium
                    break
        cost_of_equity += size_premium

        # 2. Cost of Debt
        cost_of_debt = 0.05  # Default 5%
        tax_rate = default_tax_rate

        total_debt = ticker_info.get("totalDebt")
        interest_expense = None
        income_tax_expense = None
        pretax_income = None

        if financials_df is not None and not financials_df.empty:
            latest_period = financials_df.columns[0]
            interest_expense = _get_statement_value(
                financials_df, "Interest Expense", latest_period
            )
            income_tax_expense = _get_statement_value(
                financials_df, "Tax Provision", latest_period
            )
            pretax_income = _get_statement_value(
                financials_df, "Pretax Income", latest_period
            )

            if interest_expense and total_debt and total_debt > 0:
                implied = abs(interest_expense) / total_debt
                # A tiny debt balance against a full year of interest expense
                # yields nonsense rates; clamp rather than propagate.
                cost_of_debt = float(np.clip(implied, 0.01, MAX_COST_OF_DEBT))

            if income_tax_expense and pretax_income and pretax_income > 0:
                tax_rate = float(np.clip(income_tax_expense / pretax_income, 0.0, 0.5))

        # 3. Weights
        market_cap = ticker_info.get("marketCap")
        if not market_cap:
            return {
                "wacc": float(
                    np.clip(cost_of_equity, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)
                ),
                "cost_of_equity": cost_of_equity,
                "beta": beta,
                "size_premium": size_premium,
                "method": "Cost of Equity (No Market Cap)",
            }

        total_value = market_cap + (total_debt or 0)
        weight_equity = market_cap / total_value
        weight_debt = (total_debt or 0) / total_value

        wacc = (weight_equity * cost_of_equity) + (
            weight_debt * cost_of_debt * (1 - tax_rate)
        )
        wacc = float(np.clip(wacc, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE))

        return {
            "wacc": wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "tax_rate": tax_rate,
            "weight_equity": weight_equity,
            "weight_debt": weight_debt,
            "beta": beta,
            "size_premium": size_premium,
            "method": "WACC",
        }
    except Exception as e:
        logging.error(f"Error calculating WACC: {e}")
        return {"wacc": 0.10, "method": "Default (10%) due to error"}


def estimate_growth_rate(
    financials_df: Optional[pd.DataFrame],
    ticker_info: Optional[Dict[str, Any]] = None,
    item_name: str = "Net Income",
    years: int = GROWTH_HISTORY_YEARS,
) -> float:
    """Attempts to estimate a historical growth rate for a financial item."""
    _values = []

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
                    try:
                        g_val = float(row["growth"])
                        if not np.isnan(g_val) and not np.isinf(g_val):
                            expected_rates.append(g_val)
                    except (ValueError, TypeError):
                        pass

            if expected_rates:
                avg_expected = sum(expected_rates) / len(expected_rates)
                return avg_expected

        # Fallback to standard info fields if specific estimates missing
        g = ticker_info.get("earningsGrowth") or ticker_info.get("revenueGrowth")
        if g is not None:
            try:
                g_val = float(g)
                if not np.isnan(g_val) and not np.isinf(g_val):
                    return g_val
            except (ValueError, TypeError):
                pass

    # 2. Try to calculate historical CAGR if analyst data is not available
    # ... (keeping existing historical CAGR logic as fallback)
    if financials_df is not None and not financials_df.empty:
        try:
            # Sort columns to be chronological
            cols = sorted(
                financials_df.columns, key=lambda x: pd.to_datetime(x, errors="coerce")
            )
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
                    if years_diff >= 2.5:  # approx 3 years
                        start_idx = i
                        break

                start_date, start_val = recent_dated[start_idx]

                # Calculate CAGR
                n_years = (end_date - start_date).days / 365.25
                if n_years > 0.5:
                    res = (end_val / start_val) ** (1 / n_years) - 1
                    # Protect against NaN or Infinity
                    if np.isnan(res) or np.isinf(res):
                        return 0.05
                    return res
        except Exception:
            pass

    # 3. Final Default
    return 0.05


def _historical_growth_regression(
    financials_df: Optional[pd.DataFrame],
    item_name: str = "Net Income",
    years: int = GROWTH_HISTORY_YEARS,
) -> Optional[float]:
    """Log-linear growth over the available history.

    `estimate_growth_rate` measures endpoint-to-endpoint CAGR, which reads a
    single depressed or inflated base year as a decade-long trend. Regressing
    log(value) on time uses every observation instead, so one bad year moves
    the slope a little rather than setting it.

    Returns None when there are fewer than three positive observations, which
    is the caller's signal to fall back rather than pretend to a trend.
    """
    if financials_df is None or financials_df.empty:
        return None
    try:
        dated: List[Tuple[pd.Timestamp, float]] = []
        for col in financials_df.columns:
            ts = pd.to_datetime(col, errors="coerce")
            val = _get_statement_value(financials_df, item_name, col)
            if pd.notna(ts) and val is not None and val > 0:
                dated.append((ts, val))
        dated.sort(key=lambda p: p[0])
        dated = dated[-years:]
        if len(dated) < 3:
            return None

        t0 = dated[0][0]
        xs = np.array([(d - t0).days / 365.25 for d, _ in dated], dtype=float)
        ys = np.log(np.array([v for _, v in dated], dtype=float))
        if xs[-1] - xs[0] < 1.0:
            return None

        slope = float(np.polyfit(xs, ys, 1)[0])
        growth = float(np.expm1(slope))
        return growth if np.isfinite(growth) else None
    except Exception:
        return None


def blended_growth_estimate(
    financials_df: Optional[pd.DataFrame],
    ticker_info: Optional[Dict[str, Any]] = None,
    item_name: str = "Net Income",
) -> Dict[str, Any]:
    """Forecast growth for the projection, shrunk toward a base rate.

    Separate from `estimate_growth_rate`, which stays a pure *measurement* of
    history. This is the *forecast*, and the difference matters: measured
    growth is a fact about the past, while the number a ten-year DCF needs is a
    claim about the future, and high growth mean-reverts. Feeding raw measured
    growth into a decade of compounding is what produced valuations of 10x-40000x
    price in the baseline.

    Blends whatever evidence exists (analyst near-term, log-regression trend,
    endpoint CAGR), then pulls the result most of the way to `BASE_RATE_GROWTH`
    and clamps it to a band a real business could sustain.
    """
    signals: Dict[str, float] = {}

    if ticker_info:
        analyst_rates = []
        for period in ("0y", "+1y"):
            row = (ticker_info.get("_earnings_estimate") or {}).get(period)
            if row and row.get("growth") is not None:
                try:
                    val = float(row["growth"])
                    if np.isfinite(val):
                        analyst_rates.append(val)
                except (ValueError, TypeError):
                    pass
        if analyst_rates:
            signals["analyst"] = float(np.mean(analyst_rates))

    regression = _historical_growth_regression(financials_df, item_name=item_name)
    if regression is not None:
        signals["regression"] = regression

    cagr = estimate_growth_rate(financials_df, ticker_info=None, item_name=item_name)
    # 0.05 is the estimator's "I found nothing" default; don't treat it as evidence.
    if cagr is not None and np.isfinite(cagr) and abs(cagr - 0.05) > 1e-9:
        signals["cagr"] = float(cagr)

    if not signals:
        return {
            "growth": BASE_RATE_GROWTH,
            "raw_growth": None,
            "signals": {},
            "method": "base rate (no growth evidence)",
        }

    # Combine by *kind* of evidence, not by counting estimates. `regression`
    # and `cagr` are two ways of measuring the same backward-looking quantity,
    # so a plain median over all three signals gave history two votes to the
    # analysts' one and the middle value was almost always historical — the
    # forward-looking estimate was silently discarded. Collapse history to one
    # view first, then let history and expectations weigh equally.
    historical = [signals[k] for k in ("regression", "cagr") if k in signals]
    views: List[float] = []
    if historical:
        views.append(float(np.median(historical)))
    if "analyst" in signals:
        views.append(signals["analyst"])

    raw = float(np.mean(views))

    shrunk = GROWTH_SHRINK_WEIGHT * raw + (1 - GROWTH_SHRINK_WEIGHT) * BASE_RATE_GROWTH
    clamped = float(np.clip(shrunk, MIN_PROJECTED_GROWTH, MAX_PROJECTED_GROWTH))

    return {
        "growth": clamped,
        "raw_growth": raw,
        "signals": signals,
        "method": (
            f"{sorted(signals)} -> {raw:.1%}, shrunk "
            f"{GROWTH_SHRINK_WEIGHT:.0%}/{1 - GROWTH_SHRINK_WEIGHT:.0%} toward "
            f"{BASE_RATE_GROWTH:.0%} base rate"
        ),
    }


def _enrich_ticker_info(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Fills in missing critical data in ticker_info using financial statements.
    Ensures DCF/Graham models have necessary inputs even if 'info' is sparse.
    """
    enriched = ticker_info.copy()

    # Check 1: Shares Outstanding
    if not enriched.get("sharesOutstanding"):
        # Try Balance Sheet (Ordinary Shares Number or Share Issued)
        if balance_sheet_df is not None and not balance_sheet_df.empty:
            latest_col = balance_sheet_df.columns[0]
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", latest_col
            )
            if not shares:
                shares = _get_statement_value(
                    balance_sheet_df, "Share Issued", latest_col
                )

            if shares:
                enriched["sharesOutstanding"] = shares

        # Try Income Statement (Average Shares)
        if (
            not enriched.get("sharesOutstanding")
            and financials_df is not None
            and not financials_df.empty
        ):
            latest_col = financials_df.columns[0]
            shares = _get_statement_value(
                financials_df, "Diluted Average Shares", latest_col
            )
            if not shares:
                shares = _get_statement_value(
                    financials_df, "Basic Average Shares", latest_col
                )

            if shares:
                enriched["sharesOutstanding"] = shares

    # Check 2: Total Revenue
    if (
        not enriched.get("totalRevenue")
        and financials_df is not None
        and not financials_df.empty
    ):
        latest_col = financials_df.columns[0]
        rev = _get_statement_value(financials_df, "Total Revenue", latest_col)
        if rev:
            enriched["totalRevenue"] = rev

    # Check 3: Total Cash / Debt
    if (
        enriched.get("totalCash") is None
        and balance_sheet_df is not None
        and not balance_sheet_df.empty
    ):
        latest_col = balance_sheet_df.columns[0]
        cash = _get_statement_value(
            balance_sheet_df, "Cash And Cash Equivalents", latest_col
        )
        if cash:
            enriched["totalCash"] = cash

    if (
        enriched.get("totalDebt") is None
        and balance_sheet_df is not None
        and not balance_sheet_df.empty
    ):
        latest_col = balance_sheet_df.columns[0]
        debt = _get_statement_value(balance_sheet_df, "Total Debt", latest_col)
        if debt:
            enriched["totalDebt"] = debt

    # Check 4: EPS (Trailing)
    if (
        not enriched.get("trailingEps")
        and financials_df is not None
        and not financials_df.empty
    ):
        latest_col = financials_df.columns[0]
        eps = _get_statement_value(financials_df, "Diluted EPS", latest_col)
        if eps:
            enriched["trailingEps"] = eps

    return enriched


def estimate_fcf_margin(
    financials_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    years: int = 5,
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
            key=lambda x: pd.to_datetime(x, errors="coerce"),
        )

        margins = []
        for col in common_cols[-years:]:  # Look at last N years
            rev = _get_statement_value(financials_df, "Total Revenue", col)
            ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", col)
            capex = _get_statement_value(cashflow_df, "Capital Expenditure", col)

            if rev and rev > 0 and ocf is not None and capex is not None:
                # Capex is usually negative
                fcf = ocf + capex
                margin = fcf / rev
                # Keep loss-making years. Dropping them (the old filter was
                # 0 < margin < 0.5) meant a company that burned cash in three
                # of five years was scored on its two good ones, which is the
                # single most upward-biased step in the old chain. Only
                # implausible magnitudes are excluded as data errors.
                if -1.0 < margin < 0.6:
                    margins.append(margin)

        if margins:
            # Use median for better robustness against outliers
            return float(np.median(margins))

    except Exception as e:
        logging.warning(f"Failed to estimate FCF margin: {e}")

    return 0.05  # Fallback


# A full business cycle. Five years was never a judgement about normalization —
# it was the number of annual periods yfinance returned, and it cannot contain a
# recession: a window ending in 2025 starts in 2021 and has only ever seen an
# expansion. The SEC-filed history reaches ~19 years, so the through-cycle
# figure Buffett's owner earnings actually describe is now computable.
CYCLE_YEARS = 10

# Below this the median is not a cycle, just a short average, and the shorter
# absolute-dollar path is the more honest answer.
MIN_CYCLE_OBSERVATIONS = 6


def through_cycle_fcf_margin(
    financials_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    years: int = CYCLE_YEARS,
) -> Dict[str, Any]:
    """The FCF margin this business earns in a normal year, over a full cycle.

    A *margin* rather than a dollar figure, because dollars go stale: the median
    of ten years of absolute FCF describes a company the size this one was five
    years ago, which for anything growing is not conservatism but an error. The
    margin is the durable part — it is what mean-reverts — and multiplying it by
    today's revenue puts the normalized figure back on today's scale.

    Loss years are kept. Dropping them would score a company that burned cash in
    four of ten years on the six that worked, which is the single most
    upward-biased thing a normalizer can do.
    """
    margins: List[float] = []
    try:
        if financials_df is not None and cashflow_df is not None:
            common = sorted(
                set(financials_df.columns).intersection(set(cashflow_df.columns)),
                key=lambda c: pd.to_datetime(c, errors="coerce"),
            )
            for col in common[-years:]:
                revenue = _get_statement_value(financials_df, "Total Revenue", col)
                ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", col)
                capex = _get_statement_value(cashflow_df, "Capital Expenditure", col)
                if not revenue or revenue <= 0 or ocf is None or capex is None:
                    continue
                margin = (ocf + capex) / revenue
                # The same plausibility band `estimate_fcf_margin` uses: outside
                # it the statement mapping is wrong, not the business remarkable.
                if -1.0 < margin < 0.6:
                    margins.append(margin)
    except Exception as exc:
        logging.debug(f"Through-cycle margin extraction failed: {exc}")

    if len(margins) < MIN_CYCLE_OBSERVATIONS:
        return {"margin": None, "observations": len(margins)}
    return {"margin": float(np.median(margins)), "observations": len(margins)}


def normalized_base_fcf(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    years: int = 5,
) -> Dict[str, Any]:
    """A starting cash flow that reflects the business, not one fiscal year.

    Every dollar of the projection scales linearly off this number, so using a
    single year's FCF hands a decade of value to whatever was unusual about
    that year — a big working-capital swing, a factory built, a legal
    settlement. Buffett's "owner earnings" are explicitly a normal-year figure.

    Preferred estimate: the through-cycle FCF *margin* applied to current
    revenue. It spans a recession where a five-year window cannot, and it stays
    on today's scale, which a ten-year median of absolute dollars does not.

    Falls back to the median of the last `years` absolute FCF observations when
    there is no cycle to measure — and only ever falls back, never refuses on
    the cycle's behalf: a company the short window can value is not made
    unvaluable by a longer one. Returns `fcf=None` when neither estimate finds
    positive normalized cash flow, which is a refusal, not a fallback.
    """
    history: List[float] = []
    try:
        if cashflow_df is not None and not cashflow_df.empty:
            cols = sorted(
                cashflow_df.columns, key=lambda c: pd.to_datetime(c, errors="coerce")
            )
            for col in cols[-years:]:
                ocf = _get_statement_value(cashflow_df, "Operating Cash Flow", col)
                capex = _get_statement_value(cashflow_df, "Capital Expenditure", col)
                if ocf is not None and capex is not None:
                    history.append(ocf + capex)
    except Exception as exc:
        logging.debug(f"FCF history extraction failed: {exc}")

    latest_fcf = history[-1] if history else None
    if latest_fcf is None:
        latest_fcf = ticker_info.get("freeCashflow")

    revenue = ticker_info.get("totalRevenue")
    # A reported FCF margin above 60% is almost always a bad statement mapping
    # rather than a spectacular business. Drop only the offending years: an
    # earlier version discarded the whole history whenever the *latest* year
    # looked wrong, throwing away four good observations because of one bad
    # one — the opposite of what normalizing is for.
    if revenue and revenue > 0:
        plausible = [f for f in history if f / revenue <= 0.6]
        if len(plausible) != len(history):
            logging.debug(
                f"Dropping {len(history) - len(plausible)} implausible FCF year(s) "
                f"for {ticker_info.get('symbol')}"
            )
            history = plausible
            latest_fcf = history[-1] if history else None
        if latest_fcf and latest_fcf / revenue > 0.6:
            latest_fcf = None

    # Preferred: the cycle. Positive-only, so this can rescale a valuation but
    # never take one away — a company the five-year window can value is not made
    # unvaluable by looking further back.
    cycle = through_cycle_fcf_margin(financials_df, cashflow_df)
    if cycle["margin"] and cycle["margin"] > 0 and revenue and revenue > 0:
        cycle_fcf = cycle["margin"] * revenue
        if cycle_fcf > 0:
            return {
                "fcf": float(cycle_fcf),
                "method": (
                    f"through-cycle {cycle['margin']:.1%} FCF margin over "
                    f"{cycle['observations']}y x current revenue"
                ),
                "history": history,
                "normalized": True,
            }

    if len(history) >= 3:
        median_fcf = float(np.median(history))
        if median_fcf > 0:
            # The median *is* the robust estimate — deliberately not
            # min(median, latest). Taking the lower of two estimates is a
            # haircut with no statistical basis, and such haircuts stack:
            # combined with a shrunk growth rate and a size-premium discount
            # rate it produced a model asserting the median profitable company
            # was 26% overvalued. Each conservatism was defensible alone; the
            # product of three was not.
            return {
                "fcf": median_fcf,
                "method": f"median of {len(history)}y FCF",
                "history": history,
                "normalized": True,
            }

    if (latest_fcf or 0) > 0:
        return {
            "fcf": float(latest_fcf),
            "method": "latest reported FCF (insufficient history to normalize)",
            "history": history,
            "normalized": False,
        }

    return {
        "fcf": None,
        "method": "no positive normalized free cash flow",
        "history": history,
        "normalized": False,
    }


def run_monte_carlo_dcf(
    ticker_info: Dict[str, Any],
    base_fcf: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for DCF."""
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares:
            return {}

        # 1. Generate stochastic variables
        #
        # Uncertainty on growth is *absolute*, not proportional to the estimate.
        # The old relative-20% rule gave a company forecast at 2% growth a
        # standard deviation of 0.4pp — near-certainty about the hardest number
        # in the model — while a 40%-growth forecast got 8pp. Nobody knows a
        # decade of growth to within half a point. 6pp is roughly the observed
        # cross-sectional spread of realized 5y growth around forecasts.
        growth_sigma = max(0.06, abs(base_growth) * 0.25)
        growth_samples = np.random.normal(base_growth, growth_sigma, iterations)
        # Discount rate: absolute floor on the spread too, for the same reason.
        discount_sigma = max(0.015, abs(base_discount) * 0.15)
        discount_samples = np.random.normal(base_discount, discount_sigma, iterations)

        # The band the projection is allowed to explore. Crucially the lower
        # bound is negative: the old floor of 0.0 meant no simulated future
        # ever had the business shrinking, so the "bear" percentile was not a
        # bear case at all and the distribution was biased upward by
        # construction.
        growth_samples = np.clip(
            growth_samples, MIN_PROJECTED_GROWTH, MAX_PROJECTED_GROWTH
        )
        discount_samples = np.clip(
            discount_samples, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE
        )

        # 2. Vectorized Projections
        # Shape: (iterations, projection_years)
        years = np.arange(1, projection_years + 1)

        # Linear Fade: Growth trends from growth_sample to terminal_growth over projection period
        fade_factors = (
            (years - 1) / (projection_years - 1)
            if projection_years > 1
            else np.array([0])
        )
        # yearly_growths shape: (iterations, projection_years)
        yearly_growths = (
            growth_samples[:, None]
            - (growth_samples[:, None] - terminal_growth) * fade_factors
        )

        # Calculate FCF for each year: base_fcf * cumprod(1 + g_i)
        fcf_projections = base_fcf * np.cumprod(1 + yearly_growths, axis=1)

        # 3. Present Value of FCFs
        # PV = FCF / (1 + r)^n
        pv_projections = fcf_projections / (1 + discount_samples[:, None]) ** years
        sum_pv_fcf = np.sum(pv_projections, axis=1)

        # 4. Terminal Value
        # TV = (FCF_last * (1 + g_term)) / (r - g_term)
        last_fcf = fcf_projections[:, -1]
        terminal_values = (last_fcf * (1 + terminal_growth)) / (
            discount_samples - terminal_growth
        )
        # PV of TV
        pv_terminal_values = (
            terminal_values / (1 + discount_samples) ** projection_years
        )

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
        smoothed_counts = np.convolve(counts, kernel, mode="same")

        histogram = [
            {"price": float(p), "count": float(c)}
            for p, c in zip(midpoints, smoothed_counts)
        ]

        # 7. Extract Percentiles
        # `conservative` (P25) is what the Buffett/value ranking sorts on: the
        # mean of a DCF distribution is not a price you would pay, because being
        # wrong on the downside costs more than being right on the upside.
        return {
            "bear": float(np.percentile(intrinsic_values, 10)),
            "conservative": float(np.percentile(intrinsic_values, 25)),
            "base": float(np.percentile(intrinsic_values, 50)),
            "bull": float(np.percentile(intrinsic_values, 90)),
            "std_dev": float(np.std(intrinsic_values)),
            "histogram": histogram,
        }
    except Exception as e:
        logging.error(f"Monte Carlo DCF failed: {e}")
        return {}


def run_monte_carlo_graham(
    eps: float, base_growth: float, base_bond_yield: float, iterations: int = 10000
) -> Dict[str, Any]:
    """Runs a vectorized Monte Carlo simulation for Graham's Formula."""
    try:
        # 1. Stochastic Variables
        # Growth Rate: 20% relative std dev
        growth_samples = np.random.normal(
            base_growth, abs(base_growth) * 0.2, iterations
        )
        # Bond Yield: 10% relative std dev
        yield_samples = np.random.normal(
            base_bond_yield, abs(base_bond_yield) * 0.1, iterations
        )

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
        smoothed_counts = np.convolve(counts, kernel, mode="same")

        histogram = [
            {"price": float(p), "count": float(c)}
            for p, c in zip(midpoints, smoothed_counts)
        ]

        return {
            "bear": float(np.percentile(intrinsic_values, 10)),
            "conservative": float(np.percentile(intrinsic_values, 25)),
            "base": float(np.percentile(intrinsic_values, 50)),
            "bull": float(np.percentile(intrinsic_values, 90)),
            "std_dev": float(np.std(intrinsic_values)),
            "histogram": histogram,
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
    fcf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Performs a Discounted Cash Flow (DCF) valuation.
    """
    try:
        current_revenue = ticker_info.get("totalRevenue")
        model_method = "DCF"

        # 1. Base FCF — a normalized figure, not one fiscal year.
        fcf_method = "caller-supplied FCF"
        fcf_history: List[float] = []
        if fcf is None:
            base_res = normalized_base_fcf(ticker_info, financials_df, cashflow_df)
            fcf = base_res["fcf"]
            fcf_method = base_res["method"]
            fcf_history = base_res["history"]

        # Revenue-based fallback: only when the company has a genuinely
        # profitable cash-flow history that a single bad year has interrupted.
        # The old code applied it to anything with revenue, which manufactured a
        # DCF for businesses that had never converted a dollar of sales to cash
        # (41% of all DCFs in the baseline took this path, median 0.51x price
        # and 49 of them above 10x). A company that does not generate cash is
        # not conservatively valued by assuming it soon will.
        used_fcf_margin = None
        if (fcf is None or fcf <= 0) and current_revenue and current_revenue > 0:
            if target_fcf_margin is None:
                target_fcf_margin = estimate_fcf_margin(financials_df, cashflow_df)

            positive_years = sum(1 for v in fcf_history if v > 0)
            has_track_record = (
                len(fcf_history) >= 3 and positive_years >= len(fcf_history) / 2
            )

            if target_fcf_margin > 0 and has_track_record:
                fcf = current_revenue * target_fcf_margin
                model_method = "Revenue-based DCF"
                used_fcf_margin = target_fcf_margin
                fcf_method = (
                    f"revenue x normalized {target_fcf_margin:.1%} FCF margin "
                    f"({positive_years}/{len(fcf_history)} cash-positive years)"
                )
            else:
                return {
                    "error": "No sustained positive free cash flow to value",
                    "diagnostics": {
                        "fcf_years": len(fcf_history),
                        "positive_years": positive_years,
                        "normalized_margin": target_fcf_margin,
                    },
                }

        if fcf is None or fcf <= 0:
            return {"error": "Negative or missing Free Cash Flow"}

        # 2. Discount Rate (WACC)
        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = wacc_res["wacc"]
        else:
            # Apply stability floor even to provided discount rate
            discount_rate = max(0.075, discount_rate)

        # 3. Growth Rate — shrunk toward a base rate before it is compounded.
        growth_method = "caller-supplied growth"
        growth_signals: Dict[str, float] = {}
        if growth_rate is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            growth_rate = growth_res["growth"]
            growth_method = growth_res["method"]
            growth_signals = growth_res["signals"]

        # An explicit override still gets the sanity band: no projection may
        # assume a decade at a rate no business sustains.
        applied_growth = float(
            np.clip(growth_rate, MIN_PROJECTED_GROWTH, MAX_PROJECTED_GROWTH)
        )

        projected_fcf = []
        pv_fcf = []
        current_fcf = fcf

        for y in range(1, projection_years + 1):
            # Linear Fade: Growth trends from applied_growth to terminal_rate over projection period
            fade_factor = (
                (y - 1) / (projection_years - 1) if projection_years > 1 else 0
            )
            yearly_growth = (
                applied_growth - (applied_growth - terminal_growth_rate) * fade_factor
            )

            next_fcf = current_fcf * (1 + yearly_growth)
            projected_fcf.append(next_fcf)
            pv_fcf.append(next_fcf / ((1 + discount_rate) ** y))
            current_fcf = next_fcf

        # 5. Terminal Value
        # Ensure denominator is positive
        safe_discount = max(discount_rate, terminal_growth_rate + 0.01)
        terminal_value = (current_fcf * (1 + terminal_growth_rate)) / (
            safe_discount - terminal_growth_rate
        )
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

        # Protect against NaN in final DCF result
        if np.isnan(intrinsic_value) or np.isinf(intrinsic_value):
            return {"error": "DCF resulted in invalid NaN/Inf value"}

        # A DCF whose value is almost entirely terminal is a statement about
        # the assumed perpetuity, not about the business's next decade. Surface
        # the share so the caller can weigh it instead of taking the point
        # estimate at face value.
        tv_share = pv_terminal_value / enterprise_value if enterprise_value else None

        # Net debt above enterprise value means the equity is a stub; the model
        # is not built for that and would report a negative per-share value.
        if intrinsic_value <= 0:
            return {
                "error": "Net debt exceeds discounted cash flows (no residual equity value)",
                "diagnostics": {
                    "equity_value": equity_value,
                    "enterprise_value": enterprise_value,
                },
            }

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
                "fcf_margin": used_fcf_margin,
                "fcf_method": fcf_method,
                "growth_method": growth_method,
                "growth_signals": growth_signals,
                "terminal_value_share": tv_share,
            },
        }
        notes = [f"Linear growth fade over {projection_years}y toward terminal rate"]
        if growth_rate > MAX_PROJECTED_GROWTH:
            notes.append(f"growth clamped {growth_rate:.1%} -> {applied_growth:.1%}")
        if tv_share is not None and tv_share > 0.75:
            notes.append(f"{tv_share:.0%} of value is terminal — low confidence")
        res["parameters"]["note"] = "; ".join(notes)
        return res
    except Exception as e:
        return {"error": f"DCF calculation failed: {str(e)}"}


def calculate_intrinsic_value_graham(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame] = None,
    growth_rate: Optional[float] = None,
    eps: Optional[float] = None,
    bond_yield: Optional[float] = None,
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
        growth_method = "caller-supplied growth"
        if growth_rate is None:
            # Same shrunk forecast the DCF uses, expressed in percent. Graham's
            # 'g' is explicitly a *long-run expected* rate, so feeding it raw
            # trailing growth was always a misreading of the formula.
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            growth_rate = growth_res["growth"] * 100
            growth_method = growth_res["method"]

        # Y is a corporate bond yield. Floored because the 4.4/Y term is a
        # divisor: a low yield inflates every valuation without limit.
        if bond_yield is None:
            bond_yield = RISK_FREE_RATE * 100
        bond_yield = max(bond_yield, 3.5)

        if eps is None:
            eps = ticker_info.get("trailingEps")

        book_value = None
        if balance_sheet_df is not None and not balance_sheet_df.empty:
            latest_bs_period = balance_sheet_df.columns[0]
            total_equity = _get_statement_value(
                balance_sheet_df, "Total Stockholder Equity", latest_bs_period
            ) or _get_statement_value(
                balance_sheet_df,
                "Total Equity Gross Minority Interest",
                latest_bs_period,
            )
            shares = ticker_info.get("sharesOutstanding")
            if total_equity and shares and shares > 0:
                book_value = total_equity / shares

        # No book-value substitution. This previously returned book value per
        # share *labelled as the Graham model* for 37% of the universe, so a
        # third of "Graham" valuations were a different model wearing its name,
        # then averaged in with equal weight. Book value is also the weakest
        # value signal measured on this data (see the ranking signal audit):
        # buybacks and intangibles distort it worst for exactly the durable
        # compounders this is meant to find. It is reported as context only.
        if eps is None or eps <= 0:
            return {
                "error": "Negative or missing EPS",
                "diagnostics": {"book_value_per_share": book_value},
            }

        # Graham's multiplier is unbounded in g and was written for a market
        # where 'reasonably expected' growth meant single digits. At the old
        # 30% cap the formula implied a P/E of 68 as *intrinsic* value.
        applied_growth = float(np.clip(growth_rate, -5.0, 15.0))
        intrinsic_value = (eps * (8.5 + 2 * applied_growth) * 4.4) / bond_yield

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Graham formula produced a non-positive value"}

        res = {
            "intrinsic_value": intrinsic_value,
            "model": "Graham's Revised Formula",
            "parameters": {
                "eps": eps,
                "growth_rate_pct": growth_rate,
                "applied_growth_pct": applied_growth,
                "bond_yield_proxy": bond_yield,
                "growth_method": growth_method,
                "book_value_per_share": book_value,
                "implied_pe": intrinsic_value / eps if eps else None,
            },
        }
        if growth_rate > 15.0:
            res["parameters"]["note"] = (
                "Growth capped at 15% — Graham's 'g' is a long-run rate"
            )

        return res
    except Exception as e:
        return {"error": f"Graham calculation failed: {str(e)}"}


def calculate_earnings_power_value(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    discount_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Greenwald's Earnings Power Value: what the business is worth if it never grows.

    EPV = normalized after-tax operating earnings / cost of capital, plus net cash.

    This exists to break the DCF's dependence on a growth forecast. A DCF puts
    most of its value in a terminal assumption, so two analysts with the same
    facts can differ threefold; EPV asks the narrower question "what is the
    current earning power worth in perpetuity?" and answers it from realized
    numbers only. Where EPV sits above price, the market is charging nothing
    for growth — the condition a value investor is actually hunting for.

    Uses the median of up to five years of operating income, so a single
    exceptional year does not become a perpetuity.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares or shares <= 0:
            return {"error": "Missing shares outstanding"}

        # Normalized EBIT across the cycle.
        ebit_history: List[float] = []
        if financials_df is not None and not financials_df.empty:
            cols = sorted(
                financials_df.columns, key=lambda c: pd.to_datetime(c, errors="coerce")
            )
            for col in cols[-5:]:
                ebit = _get_statement_value(financials_df, "Operating Income", col)
                if ebit is None:
                    ebit = _get_statement_value(financials_df, "EBIT", col)
                if ebit is not None:
                    ebit_history.append(ebit)

        if not ebit_history:
            return {"error": "No operating income history for earnings power"}

        normalized_ebit = float(np.median(ebit_history))
        if normalized_ebit <= 0:
            return {
                "error": "No positive normalized operating income",
                "diagnostics": {"normalized_ebit": normalized_ebit},
            }

        # Effective tax rate from the same statements, banded to sane values.
        tax_rate = 0.21
        if financials_df is not None and not financials_df.empty:
            latest = financials_df.columns[0]
            tax = _get_statement_value(financials_df, "Tax Provision", latest)
            pretax = _get_statement_value(financials_df, "Pretax Income", latest)
            if tax is not None and pretax and pretax > 0:
                tax_rate = float(np.clip(tax / pretax, 0.0, 0.5))

        if discount_rate is None:
            discount_rate = calculate_wacc(
                ticker_info, financials_df, balance_sheet_df
            )["wacc"]
        discount_rate = float(
            np.clip(discount_rate, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)
        )

        nopat = normalized_ebit * (1 - tax_rate)
        enterprise_value = nopat / discount_rate

        cash = ticker_info.get("totalCash") or 0
        debt = ticker_info.get("totalDebt") or 0
        equity_value = enterprise_value + cash - debt
        if equity_value <= 0:
            return {
                "error": "Net debt exceeds earnings power value",
                "diagnostics": {"enterprise_value": enterprise_value},
            }

        intrinsic_value = equity_value / shares
        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "EPV produced an invalid value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Earnings Power Value (no growth)",
            "parameters": {
                "normalized_ebit": normalized_ebit,
                "ebit_years": len(ebit_history),
                "tax_rate": tax_rate,
                "discount_rate": discount_rate,
                "nopat": nopat,
                "net_cash": cash - debt,
                "note": "Assumes zero growth; value of current earning power only",
            },
        }
    except Exception as e:
        return {"error": f"EPV calculation failed: {str(e)}"}


def get_comprehensive_intrinsic_value(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, Any]] = None,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """
    Consolidates multiple intrinsic value models into a single advice object.
    """
    # 0. Enrich data with statement fallbacks to maximize coverage
    ticker_info = _enrich_ticker_info(
        ticker_info, financials_df, balance_sheet_df, cashflow_df
    )

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
        ticker_info,
        financials_df,
        balance_sheet_df,
        cashflow_df,
        discount_rate=dcf_discount,
        growth_rate=dcf_growth,
        projection_years=dcf_projection,
        terminal_growth_rate=dcf_terminal,
        fcf=dcf_fcf,
        target_fcf_margin=target_fcf_margin,
    )

    graham_res = calculate_intrinsic_value_graham(
        ticker_info,
        financials_df,
        balance_sheet_df,
        growth_rate=graham_growth,
        eps=graham_eps,
        bond_yield=graham_bond_yield,
    )

    epv_res = calculate_earnings_power_value(
        ticker_info,
        financials_df,
        balance_sheet_df,
        cashflow_df,
        discount_rate=overrides.get("epv_discount_rate", dcf_discount),
    )

    dcf_mc = None
    graham_mc = None

    # Pass through iterations to MC simulations
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
            params["terminal_growth_rate"],
            iterations=iterations,
        )
        dcf_res["mc"] = dcf_mc

    if "intrinsic_value" in graham_res:
        params = graham_res["parameters"]
        if "eps" in params:
            # Use applied growth for MC if available
            mc_growth_pct = params.get(
                "applied_growth_pct", params.get("growth_rate_pct", 0)
            )
            graham_mc = run_monte_carlo_graham(
                params["eps"],
                mc_growth_pct,
                params["bond_yield_proxy"],
                iterations=iterations,
            )
            graham_res["mc"] = graham_mc

    # Simulated bear/bull bounds, collected from whichever models ran.
    bear_values: List[float] = []
    bull_values: List[float] = []
    for mc in (dcf_mc, graham_mc):
        if mc and mc.get("bear") is not None and mc.get("bull") is not None:
            bear_values.append(mc["bear"])
            bull_values.append(mc["bull"])

    current_price = ticker_info.get("currentPrice") or ticker_info.get(
        "regularMarketPrice"
    )

    # --- ETF Valuation Logic ---
    quote_type = ticker_info.get("quoteType", "").upper()
    if quote_type == "ETF" or quote_type == "MUTUALFUND":
        nav_price = ticker_info.get("navPrice")

        # If no explicit NAV, current price is the best proxy for ETFs
        if (nav_price is None or nav_price == 0) and current_price:
            nav_price = current_price

        if nav_price:
            results = {
                "current_price": current_price,
                "average_intrinsic_value": nav_price,
                "valuation_note": f"Valuation based on Net Asset Value (NAV) for {quote_type}.",
                "valuation_status": "nav",
                "models": {
                    "dcf": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "graham": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "epv": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                },
            }

            if current_price:
                mos = ((nav_price - current_price) / current_price) * 100
                if mos is not None and (np.isnan(mos) or np.isinf(mos)):
                    mos = None
                results["margin_of_safety_pct"] = mos

            return results

    results: Dict[str, Any] = {
        "current_price": current_price,
        "models": {
            "dcf": dcf_res,
            "graham": graham_res,
            "epv": epv_res,
        },
    }

    # --- Eligibility ---------------------------------------------------------
    # Refuse before blending when the inputs cannot support a valuation. The old
    # code always produced a number: 99.1% "coverage", of which 15% sat above
    # 10x or below 0.1x price. Sub-dollar and micro-cap tickers were the worst
    # (42% absurd under $1) because per-share arithmetic on a tiny denominator
    # amplifies every data error. A blank is a usable answer; a fabricated
    # fair value next to a real price is not.
    market_cap = ticker_info.get("marketCap")
    ineligible = None
    if current_price is not None and current_price < MIN_PRICE_FOR_VALUATION:
        ineligible = (
            f"Price below ${MIN_PRICE_FOR_VALUATION:.0f} — per-share valuation is "
            "dominated by data noise"
        )
    elif market_cap is not None and market_cap < MIN_MARKET_CAP_FOR_VALUATION:
        ineligible = "Market cap below $50M — fundamentals are too unreliable to value"

    if ineligible:
        results["average_intrinsic_value"] = None
        results["valuation_status"] = "ineligible"
        results["valuation_note"] = ineligible
        return results

    # --- Reliability-weighted blend -----------------------------------------
    # Replaces the previous rule, which — when two models disagreed by more
    # than 50% — discarded both and kept whichever number sat closest to the
    # current price. That rule fired on 64% of the universe and made the output
    # a function of price, so the margin of safety it produced could only ever
    # be small: the estimate was anchored to the very quantity it was supposed
    # to judge. Models are now weighted by how defensible they are, never by
    # how flattering their answer is.
    contributions: List[Dict[str, Any]] = []
    for key, model_res in (("dcf", dcf_res), ("graham", graham_res)):
        iv = model_res.get("intrinsic_value")
        if iv is not None and np.isfinite(iv) and iv > 0:
            contributions.append(
                {"key": key, "value": float(iv), "weight": MODEL_BLEND_WEIGHTS[key]}
            )

    # EPV travels alongside the estimate as the no-growth floor, not inside it.
    epv_iv = epv_res.get("intrinsic_value")
    if epv_iv is not None and np.isfinite(epv_iv) and epv_iv > 0:
        results["earnings_power_floor"] = float(epv_iv)

    if not contributions:
        reasons = [
            m.get("error") for m in (dcf_res, epv_res, graham_res) if m.get("error")
        ]
        results["average_intrinsic_value"] = None
        results["valuation_status"] = "no_model"
        results["valuation_note"] = (
            "No model could value this company: " + "; ".join(dict.fromkeys(reasons))
            if reasons
            else "No model could value this company."
        )
        return results

    total_weight = sum(c["weight"] for c in contributions)
    avg_intrinsic = sum(c["value"] * c["weight"] for c in contributions) / total_weight

    results["model_weights"] = {
        c["key"]: c["weight"] / total_weight for c in contributions
    }

    # Disagreement is information: report it rather than resolving it by fiat.
    values = [c["value"] for c in contributions]
    spread_pct = None
    if len(values) > 1 and avg_intrinsic > 0:
        spread_pct = (max(values) - min(values)) / avg_intrinsic * 100

    # --- Output sanity band --------------------------------------------------
    status = "ok"
    notes: List[str] = []
    if current_price and current_price > 0:
        ratio = avg_intrinsic / current_price
        if ratio > MAX_IV_TO_PRICE or ratio < MIN_IV_TO_PRICE:
            clamped = float(
                np.clip(
                    avg_intrinsic,
                    MIN_IV_TO_PRICE * current_price,
                    MAX_IV_TO_PRICE * current_price,
                )
            )
            notes.append(
                f"Model output {ratio:.1f}x price is outside the credible band; "
                f"clamped to {clamped / current_price:.1f}x. Treat as low confidence."
            )
            avg_intrinsic = clamped
            status = "clamped"

    if spread_pct is not None and spread_pct > 100:
        detail = ", ".join(
            "{}={:.2f}".format(c["key"], c["value"]) for c in contributions
        )
        notes.append(
            f"Models disagree by {spread_pct:.0f}% of the blended value ({detail})."
        )
        if status == "ok":
            status = "low_confidence"

    if not np.isfinite(avg_intrinsic):
        avg_intrinsic = None
        status = "no_model"

    results["average_intrinsic_value"] = avg_intrinsic
    results["valuation_status"] = status
    results["model_spread_pct"] = spread_pct
    if notes:
        results["valuation_note"] = " ".join(notes)

    # Probabilistic range from whichever simulations ran.
    if bear_values and bull_values:
        results["range"] = {
            "bear": sum(bear_values) / len(bear_values),
            "bull": sum(bull_values) / len(bull_values),
        }
        for k, v in results["range"].items():
            if v is not None and (np.isnan(v) or np.isinf(v)):
                results["range"][k] = None

    if current_price and avg_intrinsic:
        mos = ((avg_intrinsic - current_price) / current_price) * 100
        if mos is not None and (np.isnan(mos) or np.isinf(mos)):
            mos = None
        results["margin_of_safety_pct"] = mos

    return results


def get_intrinsic_value_for_symbol(
    symbol: str,
    mdp: Any,
    config_manager: Optional[Any] = None,
    iterations: int = 10000,
    force_refresh: bool = False,
    prefetched_financials: Optional[pd.DataFrame] = None,
    prefetched_balance_sheet: Optional[pd.DataFrame] = None,
    prefetched_cashflow: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Higher-level helper to calculate intrinsic value with full data and overrides.
    Ensures consistency across API, scripts, and workers.

    CRITICAL: This always fetches complete statements to ensure high-quality calculations.
    """
    # Use local imports to avoid potential circular dependencies
    from finutils import map_to_yf_symbol
    import config

    # 1. Map symbol
    user_symbol_map = {}
    user_excluded_symbols = set()

    if config_manager:
        user_symbol_map = config_manager.manual_overrides.get("user_symbol_map", {})
        user_excluded_symbols = set(
            config_manager.manual_overrides.get("user_excluded_symbols", [])
        )
    else:
        # Fallback to config defaults if no manager provided
        user_symbol_map = getattr(config, "SYMBOL_MAP_TO_YFINANCE", {})
        user_excluded_symbols = set(getattr(config, "YFINANCE_EXCLUDED_SYMBOLS", []))

    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
            return {"error": f"Symbol {symbol} is in the exclusion list."}
        return {"error": f"Could not map {symbol} to Yahoo Finance symbol"}

    # 2. Fetch COMPLETE data
    # This fulfills the USER requirement: "Always fetch complete statements"
    try:
        info = mdp.get_fundamental_data(yf_symbol, force_refresh=force_refresh)

        # CRITICAL: Detect "poisoned" or insufficient info and force fresh fetch
        # If it's an Equity, we expect more identifiers.
        is_sparse = not info or len(info) <= 8
        if info and info.get("quoteType", "").upper() == "EQUITY":
            if not info.get("lastFiscalYearEnd") or not info.get("mostRecentQuarter"):
                is_sparse = True

        if is_sparse:
            logging.warning(
                f"Detected sparse/poisoned fundamental info for {yf_symbol} in get_intrinsic_value_for_symbol. Forcing refresh."
            )
            info = mdp.get_fundamental_data(yf_symbol, force_refresh=True)

        if not info:
            return {"error": f"No fundamental data found for {yf_symbol}"}

        # Patch with live price for accuracy (fundamentals cache can be up to 24h)
        try:
            # We use mdp to get a fresh quote (1-min cache)
            q_res, _, _, _, _ = mdp.get_current_quotes(
                [symbol],
                {getattr(config, "DEFAULT_CURRENCY", "USD")},
                user_symbol_map,
                user_excluded_symbols,
            )
            if symbol in q_res:
                live_p = q_res[symbol].get("price")
                if live_p:
                    info["regularMarketPrice"] = live_p
                    info["currentPrice"] = live_p
        except Exception as e_live:
            logging.warning(
                f"Live price patch failed during intrinsic value calculation for {symbol}: {e_live}"
            )

        # Use Prefetched or Fetch New
        financials = (
            prefetched_financials
            if prefetched_financials is not None
            else mdp.get_financials(yf_symbol, "annual", force_refresh=force_refresh)
        )
        balance_sheet = (
            prefetched_balance_sheet
            if prefetched_balance_sheet is not None
            else mdp.get_balance_sheet(yf_symbol, "annual", force_refresh=force_refresh)
        )
        cashflow = (
            prefetched_cashflow
            if prefetched_cashflow is not None
            else mdp.get_cashflow(yf_symbol, "annual", force_refresh=force_refresh)
        )

        # 3. Handle Overrides
        symbol_overrides = {}
        if config_manager:
            val_overrides = config_manager.manual_overrides.get(
                "valuation_overrides", {}
            )
            symbol_overrides = val_overrides.get(yf_symbol.upper(), {})
            if not symbol_overrides:
                # Try mapped symbol (original)
                symbol_overrides = val_overrides.get(symbol.upper(), {})

        # 4. Calculate
        results = get_comprehensive_intrinsic_value(
            info,
            financials,
            balance_sheet,
            cashflow,
            overrides=symbol_overrides,
            iterations=iterations,
        )

        return results
    except Exception as e:
        logging.error(f"Error in get_intrinsic_value_for_symbol for {yf_symbol}: {e}")
        return {"error": str(e)}
