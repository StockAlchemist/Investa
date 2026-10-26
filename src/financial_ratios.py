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

import contextlib
import hashlib

import pandas as pd
import investa_core
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

# --- The central blend ------------------------------------------------------
#
# Prior weights, before each model's own uncertainty is applied. Membership is
# chosen by what the business *is*, because a model that cannot describe the
# company does not become informative by being averaged in:
#
#  - A bank has no meaningful capex or free cash flow, so the DCF is not a
#    weaker view of it — it is not a view of it at all. Financials are valued
#    on discounted net income, the same rule the best-fit framework already
#    states and the blend used to ignore.
#  - EPV stays out. It is the value of current earning power with no growth —
#    a *floor*, reported beside the estimate as `earnings_power_floor`.
#  - The price multiples (mean P/E, P/B, P/S) stay out because they are anchored
#    to the current price: with no historical multiple to hand they fall back to
#    the *trailing* multiple, so intrinsic value collapses to price exactly. A
#    model that returns the number it is meant to judge cannot vote on it.
#  - A REIT reports net income after a large non-cash depreciation charge, so
#    every earnings model understates it — the two that describe it are cash
#    from operations (the closest thing here to FFO) and the dividend, which a
#    REIT is required to pay out of ~90% of taxable income.
#  - Graham's formula is not applied to financials: it prices an industrial
#    earnings stream against a bond yield and has no notion of leverage or
#    regulatory capital, which is why it calls the median bank 42% cheap.
MODEL_BLEND_PRIORS_OPERATING = {"dcf": 0.55, "graham": 0.30, "ddm": 0.15}
MODEL_BLEND_PRIORS_FINANCIAL = {"dni": 0.85, "ddm": 0.15}
MODEL_BLEND_PRIORS_REIT = {"dcfo": 0.50, "ddm": 0.50}

# The DDM values the *dividend*, not the company: for a business paying out a
# third of its earnings it returns roughly a third of the value, and blending it
# at a fixed weight for every dividend payer was the single largest calibration
# defect in the previous blend. Measured over the cached universe, DDM margin of
# safety tracks payout ratio at rho=+0.64 (median MOS -83% below a quarter
# payout, -38% above 80%) while the DCF's is flat across the same buckets. It
# only describes the whole company when nearly all distributable cash is paid
# out; below that it is reported as `dividend_discount_floor` instead. REITs are
# exempt: they distribute by statute, and their payout ratio reads above 1.0
# only because the denominator is depreciation-charged net income.
DDM_MIN_PAYOUT_FOR_BLEND = 0.60

# Each model ships a Monte Carlo band already. Turning that band into weight is
# nearly free and it is the honest answer to "how much do I trust this model for
# *this* company": cv = (p90 - p10) / (2 x p50), so a model whose own simulation
# spans a factor of two is worth less than one that spans ten percent. The floor
# keeps a wide-but-relevant model from being silenced entirely.
RELIABILITY_CV_REF = 0.60
RELIABILITY_FLOOR = 0.25

# Historical valuation multiples. The market's own record of what it has paid
# for this company's earnings, book and sales, period by period, priced at each
# fiscal period end. A *median* over that record, not a mean: one trough-earnings
# year can put a P/E in the hundreds without saying anything about the normal
# multiple. Observations outside the bands are dropped as data errors rather
# than clipped, and below the minimum count the model refuses — the alternative
# was falling back to the trailing multiple, which made intrinsic value equal
# price exactly.
HISTORICAL_MULTIPLE_MIN_OBSERVATIONS = 4
HISTORICAL_MULTIPLE_BANDS = {
    "pe": (2.0, 80.0),
    "pb": (0.15, 20.0),
    "ps": (0.05, 30.0),
}

# Below this the estimate is published as low confidence. Calibrated so that the
# old ">100% disagreement" rule still lands on the wrong side of it.
LOW_CONFIDENCE_THRESHOLD = 0.40

# Canonical accounting statement aliases mapping GAAP/IFRS variations
STATEMENT_ALIASES: Dict[str, List[str]] = {
    "Total Revenue": [
        "Total Revenue",
        "Operating Revenue",
        "Total Operating Income As Reported",
        "Revenue",
        "Gross Revenue",
    ],
    "Operating Income": [
        "Operating Income",
        "Operating Profit",
        "Total Operating Income As Reported",
        "EBIT",
        "Ebit",
    ],
    "Net Income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations",
        "Net Income From Continuing Operation Net Minority Interest",
        "Net Income Including Noncontrolling Interests",
    ],
    "Operating Cash Flow": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Cash Flowsfromusedin Operating Activities Direct",
        "Total Cash From Operating Activities",
    ],
    "Capital Expenditure": [
        "Capital Expenditure",
        "Capital Expenditure Reported",
        "Purchase Of PPE",
        "Payments For Property Plant And Equipment",
        "Net PPE Purchase And Sale",
    ],
    "Free Cash Flow": ["Free Cash Flow"],
    "Cash Dividends Paid": [
        "Cash Dividends Paid",
        "Common Stock Dividend Paid",
        "Payment Of Dividends",
        "Total Cash Dividends Paid",
    ],
    "Tax Provision": [
        "Tax Provision",
        "Provision For Income Taxes",
        "Income Tax Expense",
        "Tax Rate For Calcs",
    ],
    "Pretax Income": [
        "Pretax Income",
        "Income Before Tax",
        "Earnings Before Taxes",
        "Pretax Operating Income",
    ],
    "Total Stockholder Equity": [
        "Total Stockholder Equity",
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
        "Total Stockholders' Equity",
    ],
    "Total Cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash Financial",
        "Cash Cash Equivalents And Federal Funds Sold",
        "Cash Equivalents",
    ],
    "Total Debt": [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
        "Current Debt And Capital Lease Obligation",
    ],
    "Diluted EPS": [
        "Diluted EPS",
        "Basic EPS",
        "Diluted Normalized EPS",
        "EPS Diluted",
    ],
    "Interest Expense": [
        "Interest Expense",
        "Interest Expense Non Operating",
        "Total Interest Expense",
        "Interest Paid Supplemental Data",
    ],
    "Ordinary Shares Number": [
        "Ordinary Shares Number",
        "Share Issued",
        "Diluted Average Shares",
        "Basic Average Shares",
    ],
}


def _get_statement_value(
    df: Optional[pd.DataFrame], item_name: str, period_column: str
) -> Optional[float]:
    """Safely retrieves a value from a financial statement DataFrame with alias fallback."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_get_statement_value(df, item_name, period_column)


def _extract_fcf_from_statement(
    cf_df: Optional[pd.DataFrame], col: Any
) -> Optional[float]:
    """Extracts Free Cash Flow from cashflow statement using explicit FCF or OCF + CapEx."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_extract_fcf_from_statement(cf_df, col)


# A share count is a level, not a flow: four quarters of "Diluted Average
# Shares" summed would report four times the company's shares. These rows are
# averaged over the window instead of added.
_LEVEL_ROW_MARKERS = ("Shares", "Share Issued")


def _is_level_row(label: Any) -> bool:
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_is_level_row(label)


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
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_is_consecutive_window(newest, oldest, quarters)


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
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_to_trailing_twelve_months(statement_df, min_quarters)


def _price_on_or_before(prices_data: Any, dt: Any) -> Optional[float]:
    """Finds the split-consistent price on or immediately before `dt`."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_price_on_or_before(prices_data, dt)


def _opt_number(value: Any) -> Optional[float]:
    """A statement or profile number for Rust: None stays None; ints, floats,
    bools and NumPy numbers convert; anything else raises TypeError, as the
    Python arithmetic it replaces would have."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_opt_number(value)


def _scalar_projection(
    base: Any,
    applied_growth: Any,
    discount_rate: Any,
    projection_years: Any,
    terminal_growth_rate: Any,
) -> Tuple[Any, Any, Any]:
    """The models' faded projection with mid-year discounting and a Gordon
    terminal value: (sum of discounted flows, last flow, discounted terminal).

    Computed in Rust (investa_core.scalar_projection) with Python's scalar
    semantics. When any input is a NumPy scalar the original's terms were NumPy
    scalars too, which changes how sum() adds them and what type comes back,
    so both are mirrored.
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_scalar_projection(
        base, applied_growth, discount_rate, projection_years, terminal_growth_rate
    )


def _key_ratios_python(
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
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_key_ratios_python(
        financials_df, balance_sheet_df, cashflow_df, periods_per_year, prices_df
    )


# Statement lookups the ratio history makes, by investa_core.key_ratios slot:
# (statement, label, period). "fin"/"bs"/"cf" read this period's column,
# "prev_fin"/"prev_bs" the column a year earlier.
_RATIO_LOOKUPS = [
    ("fin", "Total Revenue"), ("fin", "Cost Of Revenue"), ("fin", "Gross Profit"),
    ("fin", "Operating Income"), ("fin", "Operating Revenue Or Loss"), ("fin", "Ebit"),
    ("fin", "Net Income"), ("fin", "Net Income From Continuing Ops"), ("fin", "Interest Expense"),
    ("fin", "Pretax Income"), ("fin", "Tax Provision"), ("fin", "Diluted Average Shares"),
    ("fin", "Diluted EPS"), ("fin", "EBITDA"), ("fin", "Ebitda"), ("fin", "Normalized EBITDA"),
    ("fin", "Reconciled Depreciation"), ("fin", "Depreciation And Amortization"),
    ("bs", "Stockholders Equity"), ("bs", "Total Stockholder Equity"),
    ("bs", "Total Equity Gross Minority Interest"), ("bs", "Total Assets"),
    ("bs", "Current Assets"), ("bs", "Total Current Assets"), ("bs", "Current Liabilities"),
    ("bs", "Total Current Liabilities"), ("bs", "Inventory"),
    ("bs", "Total Liabilities Net Minority Interest"), ("bs", "Total Liab"),
    ("bs", "Total Liabilities"), ("bs", "Long Term Debt"),
    ("bs", "Long Term Debt And Capital Lease Obligation"), ("bs", "Long Term Debt Non Current"),
    ("bs", "Total Debt"), ("bs", "Cash And Cash Equivalents"), ("bs", "Ordinary Shares Number"),
    ("cf", "Operating Cash Flow"), ("cf", "Capital Expenditure"),
    ("cf", "Common Stock Dividend Paid"), ("cf", "Cash Dividends Paid"),
    ("cf", "Depreciation And Amortization"), ("cf", "Depreciation Amortization Depletion"),
    ("cf", "Depreciation"),
    ("prev_bs", "Total Stockholder Equity"), ("prev_bs", "Total Assets"),
    ("prev_fin", "Total Revenue"), ("prev_fin", "Diluted EPS"),
]  # fmt: skip
# The fields the original may leave as None (the rest are np.nan when missing).
_RATIO_OPTIONAL = {
    "Total Revenue",
    "Operating Income",
    "Net Income",
    "Diluted EPS",
    "Price",
}
_RATIO_FIELDS = investa_core.key_ratio_fields()


def _numeric_statement(df: Optional[pd.DataFrame]) -> bool:
    """Whether every lookup in `df` can be resolved up front.

    `_get_statement_value` reads one cell at a time, and the original only
    reads some cells under conditions (a price, a positive share count). When
    every cell is a number or missing, no read can fail, so reading them all at
    once is the same as reading them in the original's order. Duplicate labels
    or columns make `.loc` return a Series, and text cells can make `float()`
    raise; those frames keep the original code.
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_numeric_statement(df)


# `_get_statement_value` for one numeric frame, resolved a row at a time: the
# label's value, else the first of its aliases that has one.
# Ported to Rust: rust/investa_core/src/fr.rs
_StatementLookup = investa_core.StatementLookup


def _period_columns(df: pd.DataFrame) -> List[Tuple[Any, Any]]:
    """(parsed date, column) in column order, parsed one column at a time as the
    original's `pd.to_datetime(col, errors="coerce") == period` does."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_period_columns(df)


def _first_column(columns: List[Tuple[Any, Any]], period: Any) -> Any:
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_first_column(columns, period)


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

    Each statement lookup is resolved once per frame and the per-period
    arithmetic runs in investa_core.key_ratios; frames with duplicate labels
    or non-numeric cells take the original code (`_key_ratios_python`).
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_key_ratios_timeseries(
        financials_df, balance_sheet_df, cashflow_df, periods_per_year, prices_df
    )


_HISTORICAL_MULTIPLE_COLUMNS = {
    "pe": "P/E Ratio",
    "pb": "P/B Ratio",
    "ps": "P/S Ratio",
}


def historical_mean_multiples(
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame] = None,
    prices_df: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """What the market has historically paid for this company's fundamentals.

    Each fiscal period is priced with the quote as of that period end, giving one
    P/E, P/B and P/S observation per year; the median across them is the multiple
    the model applies. That is a fact about the company's own trading history —
    "it usually changes hands at 14x and is at 22x" — as distinct from the
    trailing multiple, which is today's price divided by today's earnings and so
    carries no information a price does not already carry.

    Returns `{}` for any multiple without enough usable history. Callers must
    treat that as a refusal, not as a cue to substitute the current multiple.
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_historical_mean_multiples(
        financials_df, balance_sheet_df, cashflow_df, prices_df
    )


def load_price_history_for_multiples(
    mdp: Any, yf_symbol: str, years: int = 15
) -> Optional[Any]:
    """Daily closes far enough back to median a multiple across a cycle.

    Best-effort: the historical multiples are the only thing that needs it, and
    those models refuse cleanly when it is missing.
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_load_price_history_for_multiples(mdp, yf_symbol, years)


def calculate_current_valuation_ratios(
    ticker_info: Optional[Dict],
    financials_df_latest_annual: Optional[pd.DataFrame] = None,
    balance_sheet_df_latest_annual: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[float]]:
    """Calculates point-in-time valuation ratios."""
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_current_valuation_ratios(
        ticker_info, financials_df_latest_annual, balance_sheet_df_latest_annual
    )


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

    Cost of equity is CAPM with a Blume-adjusted beta plus a size premium; cost
    of debt is interest / debt, or a coverage-based synthetic spread; weights
    are at market value. The inputs are read here and the arithmetic runs in
    Rust (investa_core.wacc).
    """
    # Ported to Rust: rust/investa_core/src/fr.rs
    return investa_core.fr_calculate_wacc(
        ticker_info,
        financials_df,
        balance_sheet_df,
        risk_free_rate,
        market_return,
        default_tax_rate,
    )


def estimate_growth_rate(
    financials_df: Optional[pd.DataFrame],
    ticker_info: Optional[Dict[str, Any]] = None,
    item_name: str = "Net Income",
    years: int = GROWTH_HISTORY_YEARS,
) -> float:
    """Attempts to estimate a historical growth rate for a financial item."""
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_estimate_growth_rate(
        financials_df, ticker_info, item_name, years
    )


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
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_historical_growth_regression(financials_df, item_name, years)


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
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_blended_growth_estimate(
        financials_df, ticker_info, item_name
    )


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
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_enrich_ticker_info(
        ticker_info, financials_df, balance_sheet_df, cashflow_df
    )


def estimate_fcf_margin(
    financials_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    years: int = 5,
) -> float:
    """
    Estimates a normalized Free Cash Flow margin based on historical data.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_estimate_fcf_margin(financials_df, cashflow_df, years)


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
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_through_cycle_fcf_margin(financials_df, cashflow_df, years)


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
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_normalized_base_fcf(
        ticker_info, financials_df, cashflow_df, years
    )


def _generate_mc_stats(intrinsic_values: np.ndarray) -> Dict[str, Any]:
    """Percentiles, standard deviation and a 40-bin smoothed histogram of the
    positive, finite Monte Carlo samples; {} when fewer than 50 remain.

    Computed in Rust (investa_core.mc_stats). The smoothing stays np.convolve,
    whose BLAS dot product cannot be reproduced bit for bit.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_generate_mc_stats(intrinsic_values)


def run_monte_carlo_dcf(
    ticker_info: Dict[str, Any],
    base_fcf: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Discounted Free Cash Flow (DCF): equity value per share.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core.mc_dcf). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_dcf(
        ticker_info, base_fcf, base_growth, base_discount,
        projection_years, terminal_growth, iterations,
    )  # fmt: skip


def run_monte_carlo_dcfo(
    ticker_info: Dict[str, Any],
    base_cfo: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Discounted Cash from Operations (D-CFO), per share.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core.mc_discounted). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_dcfo(
        ticker_info, base_cfo, base_growth, base_discount,
        projection_years, terminal_growth, iterations,
    )  # fmt: skip


def run_monte_carlo_dni(
    base_eps: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Discounted Net Income (D-NI), per share.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core.mc_discounted). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_dni(
        base_eps,
        base_growth,
        base_discount,
        projection_years,
        terminal_growth,
        iterations,
    )


def run_monte_carlo_mean_pe(
    eps: float,
    applied_pe: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Mean P/E valuation: sampled EPS x sampled P/E.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_mean_pe(eps, applied_pe, iterations)


def run_monte_carlo_peg(
    eps: float,
    applied_growth_pct: float,
    dividend_yield_pct: float = 0.0,
    target_peg: float = 1.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo PEG fair value: EPS x clip(PEG x (growth + yield), 5, 35).

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_peg(
        eps, applied_growth_pct, dividend_yield_pct, target_peg, iterations
    )


def run_monte_carlo_mean_pb(
    book_value_per_share: float,
    applied_pb: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Mean P/B valuation: sampled book value x sampled P/B.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_mean_pb(book_value_per_share, applied_pb, iterations)


def run_monte_carlo_mean_ps(
    sales_per_share: float,
    applied_ps: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Mean P/S valuation: sampled sales x sampled P/S.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_mean_ps(sales_per_share, applied_ps, iterations)


def run_monte_carlo_psg(
    sales_per_share: float,
    applied_growth_pct: float,
    gross_margin: float,
    target_psg: float = 1.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Price-to-Sales Growth (PSG): sales per share x
    clip(growth/10 x margin x PSG x 1.5, 0.5, 15).

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_psg(
        sales_per_share, applied_growth_pct, gross_margin, target_psg, iterations
    )


def run_monte_carlo_graham(
    eps: float, base_growth: float, base_bond_yield: float, iterations: int = 10000
) -> Dict[str, Any]:
    """Monte Carlo Graham formula: EPS x (8.5 + 2g) x 4.4 / bond yield.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_graham(eps, base_growth, base_bond_yield, iterations)


def run_monte_carlo_ddm(
    base_dividend: float,
    base_growth: float,
    base_discount: float,
    projection_years: int = 10,
    terminal_growth: float = 0.02,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Dividend Discount Model (DDM), per share.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core.mc_discounted). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_ddm(
        base_dividend,
        base_growth,
        base_discount,
        projection_years,
        terminal_growth,
        iterations,
    )


def run_monte_carlo_epv(
    ticker_info: Dict[str, Any],
    normalized_ebit: float,
    tax_rate: float,
    base_discount: float,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Earnings Power Value (EPV): NOPAT / discount, plus net
    cash, per share, floored at 0.01.

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_epv(
        ticker_info, normalized_ebit, tax_rate, base_discount, iterations
    )


def run_monte_carlo_lynch(
    eps: float,
    growth_rate_pct: float,
    dividend_yield_pct: float = 0.0,
    iterations: int = 10000,
) -> Dict[str, Any]:
    """Monte Carlo Peter Lynch fair value: EPS x clip(growth + yield, 5, 25).

    The draws are NumPy's, in the original order; the per-sample valuation
    runs in Rust (investa_core). Original frozen in
    tests/reference/monte_carlo.py.
    """
    # Ported to Rust: rust/investa_core/src/fr_growth.rs
    return investa_core.fr_mc_lynch(
        eps, growth_rate_pct, dividend_yield_pct, iterations
    )


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
    Performs a Discounted Cash Flow (DCF) valuation with mid-year discounting.
    """
    try:
        current_revenue = ticker_info.get("totalRevenue")
        if (
            not current_revenue
            and financials_df is not None
            and not financials_df.empty
        ):
            current_revenue = _get_statement_value(
                financials_df, "Total Revenue", financials_df.columns[0]
            )
        model_method = "DCF"

        # 1. Base FCF — a normalized figure, not one fiscal year.
        fcf_method = "caller-supplied FCF"
        fcf_history: List[float] = []
        if fcf is None:
            base_res = normalized_base_fcf(ticker_info, financials_df, cashflow_df)
            fcf = base_res["fcf"]
            fcf_method = base_res["method"]
            fcf_history = base_res["history"]

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
            discount_rate = max(0.075, discount_rate)

        # 3. Growth Rate
        growth_method = "caller-supplied growth"
        growth_signals: Dict[str, float] = {}
        if growth_rate is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            growth_rate = growth_res["growth"]
            growth_method = growth_res["method"]
            growth_signals = growth_res["signals"]

        applied_growth = float(
            np.clip(growth_rate, MIN_PROJECTED_GROWTH, MAX_PROJECTED_GROWTH)
        )

        # 4-5. Linear growth fade with mid-year discounting (cash flows arrive
        # continuously across the year), then a Gordon terminal value.
        sum_pv_fcf, _last_fcf, pv_terminal_value = _scalar_projection(
            fcf, applied_growth, discount_rate, projection_years, terminal_growth_rate
        )

        # 6. Enterprise Value to Equity Value
        enterprise_value = sum_pv_fcf + pv_terminal_value
        cash = ticker_info.get("totalCash")
        if cash is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            cash = _get_statement_value(
                balance_sheet_df, "Total Cash", balance_sheet_df.columns[0]
            )
        cash = cash or 0

        debt = ticker_info.get("totalDebt")
        if debt is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            debt = _get_statement_value(
                balance_sheet_df, "Total Debt", balance_sheet_df.columns[0]
            )
        debt = debt or 0

        equity_value = enterprise_value + cash - debt

        shares_outstanding = ticker_info.get("sharesOutstanding")
        if (
            not shares_outstanding
            and balance_sheet_df is not None
            and not balance_sheet_df.empty
        ):
            shares_outstanding = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )
        if not shares_outstanding:
            return {"error": "Missing shares outstanding"}

        intrinsic_value = equity_value / shares_outstanding

        if np.isnan(intrinsic_value) or np.isinf(intrinsic_value):
            return {"error": "DCF resulted in invalid NaN/Inf value"}

        tv_share = pv_terminal_value / enterprise_value if enterprise_value else None

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
                "total_cash": cash,
                "total_debt": debt,
                "shares_outstanding": shares_outstanding,
                "fcf_margin": used_fcf_margin,
                "fcf_method": fcf_method,
                "growth_method": growth_method,
                "growth_signals": growth_signals,
                "terminal_value_share": tv_share,
                "mid_year_discounting": True,
            },
        }
        notes = [
            f"Linear growth fade over {projection_years}y with mid-year discounting"
        ]
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
        if eps is None and financials_df is not None and not financials_df.empty:
            eps = _get_statement_value(
                financials_df, "Diluted EPS", financials_df.columns[0]
            )

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
            if (
                not shares
                and balance_sheet_df is not None
                and not balance_sheet_df.empty
            ):
                shares = _get_statement_value(
                    balance_sheet_df, "Ordinary Shares Number", latest_bs_period
                )
            if total_equity and shares and shares > 0:
                book_value = total_equity / shares

        # No book-value substitution. This previously returned book value per
        # share *labelled as the Graham model* for 37% of the universe.
        # Book value is reported as context and used for Graham Number.
        if eps is None or eps <= 0:
            return {
                "error": "Negative or missing EPS",
                "diagnostics": {"book_value_per_share": book_value},
            }

        applied_growth = float(np.clip(growth_rate, -5.0, 15.0))
        intrinsic_value = (eps * (8.5 + 2 * applied_growth) * 4.4) / bond_yield

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Graham formula produced a non-positive value"}

        graham_number = (
            float(np.sqrt(22.5 * eps * book_value))
            if (book_value and book_value > 0 and eps and eps > 0)
            else None
        )

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
                "graham_number": graham_number,
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


def calculate_intrinsic_value_ddm(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    discount_rate: Optional[float] = None,
    growth_rate: Optional[float] = None,
    projection_years: int = 10,
    terminal_growth_rate: float = 0.02,
    base_dividend: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates intrinsic value using a Multi-Stage Dividend Discount Model (DDM).
    Ideal for dividend-paying companies, financials, banks, insurance, and REITs.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )

        current_price = ticker_info.get("currentPrice") or ticker_info.get(
            "regularMarketPrice"
        )

        # 1. Base Dividend per share
        if base_dividend is None:
            base_dividend = ticker_info.get("dividendRate") or ticker_info.get(
                "trailingAnnualDividendRate"
            )

        if (
            (base_dividend is None or base_dividend <= 0)
            and cashflow_df is not None
            and not cashflow_df.empty
            and shares
            and shares > 0
        ):
            latest_col = cashflow_df.columns[0]
            div_paid = _get_statement_value(
                cashflow_df, "Cash Dividends Paid", latest_col
            )
            if div_paid and div_paid != 0:
                base_dividend = abs(div_paid) / shares

        if base_dividend is None or base_dividend <= 0:
            return {"error": "No positive dividend rate reported"}

        # Dividend sustainability check
        payout = ticker_info.get("payoutRatio")
        if payout is not None and payout > 1.5:
            return {
                "error": f"Dividend payout ratio ({payout:.1%}) exceeds sustainable threshold (>150%)",
                "diagnostics": {"payout_ratio": payout, "base_dividend": base_dividend},
            }

        # 2. Dividend Growth Rate
        growth_method = "caller-supplied dividend growth"
        if growth_rate is None:
            div_history = []
            if cashflow_df is not None and not cashflow_df.empty:
                cols = sorted(
                    cashflow_df.columns,
                    key=lambda c: pd.to_datetime(c, errors="coerce"),
                )
                for c in cols[-10:]:
                    d = _get_statement_value(cashflow_df, "Cash Dividends Paid", c)
                    if d is not None and d != 0:
                        div_history.append(abs(d))

            if len(div_history) >= 3:
                first_d, last_d = div_history[0], div_history[-1]
                n_yrs = len(div_history) - 1
                if first_d > 0 and last_d > 0 and n_yrs > 0:
                    hist_cagr = (last_d / first_d) ** (1.0 / n_yrs) - 1.0
                    raw_growth = hist_cagr if np.isfinite(hist_cagr) else 0.035
                    growth_method = f"{len(div_history)}y historical dividend CAGR ({raw_growth:.1%})"
                else:
                    raw_growth = 0.035
            else:
                growth_res = blended_growth_estimate(
                    financials_df, ticker_info=ticker_info, item_name="Net Income"
                )
                raw_growth = growth_res["growth"]
                growth_method = f"Earnings growth blend ({raw_growth:.1%})"

            # Shrink toward sustainable long-run dividend growth (3.5%)
            shrunk = 0.35 * raw_growth + 0.65 * 0.035
            applied_growth = float(np.clip(shrunk, -0.02, 0.12))
        else:
            applied_growth = float(np.clip(growth_rate, -0.05, 0.15))

        # 3. Cost of Equity (CAPM)
        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = (
                wacc_res.get("cost_of_equity") or wacc_res.get("wacc") or 0.09
            )
        discount_rate = float(
            np.clip(discount_rate, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)
        )

        # 4-5. Projections with mid-year discounting, Gordon terminal value
        sum_pv_divs, _last_div, pv_terminal_value = _scalar_projection(
            base_dividend,
            applied_growth,
            discount_rate,
            projection_years,
            terminal_growth_rate,
        )

        intrinsic_value = sum_pv_divs + pv_terminal_value

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "DDM produced non-positive value"}

        tv_share = pv_terminal_value / intrinsic_value if intrinsic_value else None

        res = {
            "intrinsic_value": intrinsic_value,
            "model": "Dividend Discount Model (Multi-Stage)",
            "parameters": {
                "base_dividend": base_dividend,
                "dividend_yield_pct": (base_dividend / current_price) * 100
                if current_price and current_price > 0
                else None,
                "payout_ratio": payout,
                "growth_rate": growth_rate
                if growth_rate is not None
                else applied_growth,
                "applied_growth": applied_growth,
                "terminal_growth_rate": terminal_growth_rate,
                "discount_rate": discount_rate,
                "projection_years": projection_years,
                "growth_method": growth_method,
                "terminal_value_share": tv_share,
                "mid_year_discounting": True,
            },
        }
        notes = [
            f"Multi-stage DDM fading to {terminal_growth_rate:.1%} terminal growth"
        ]
        if tv_share is not None and tv_share > 0.75:
            notes.append(f"{tv_share:.0%} of value is terminal")
        res["parameters"]["note"] = "; ".join(notes)
        return res
    except Exception as e:
        return {"error": f"DDM calculation failed: {str(e)}"}


def calculate_intrinsic_value_lynch(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    growth_rate: Optional[float] = None,
    eps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates Peter Lynch Fair Value based on the principle that fair P/E equals growth rate (PEG=1.0),
    adjusted for dividend yield: Fair Value = EPS * min(max(Growth + DivYield, 5), 25).
    """
    try:
        if eps is None:
            eps = ticker_info.get("trailingEps")
        if eps is None and financials_df is not None and not financials_df.empty:
            eps = _get_statement_value(
                financials_df, "Diluted EPS", financials_df.columns[0]
            )

        if eps is None or eps <= 0:
            return {"error": "Negative or missing EPS"}

        if growth_rate is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            growth_rate = growth_res["growth"] * 100

        div_yield_pct = 0.0
        div_rate = ticker_info.get("dividendRate") or ticker_info.get(
            "trailingAnnualDividendRate"
        )
        current_price = ticker_info.get("currentPrice") or ticker_info.get(
            "regularMarketPrice"
        )
        if div_rate and current_price and current_price > 0:
            div_yield_pct = (div_rate / current_price) * 100
        elif ticker_info.get("dividendYield"):
            div_yield_pct = ticker_info["dividendYield"] * 100

        # Peter Lynch fair multiplier = growth + dividend yield, clamped to sane range [5.0, 25.0]
        fair_multiplier = float(np.clip(growth_rate + div_yield_pct, 5.0, 25.0))
        intrinsic_value = eps * fair_multiplier

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Lynch formula produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Peter Lynch Fair Value",
            "parameters": {
                "eps": eps,
                "growth_rate_pct": growth_rate,
                "dividend_yield_pct": div_yield_pct,
                "fair_pe_multiplier": fair_multiplier,
                "note": "Fair Value where PEG = 1.0 (adjusted for dividend yield)",
            },
        }
    except Exception as e:
        return {"error": f"Lynch calculation failed: {str(e)}"}


def calculate_earnings_power_value(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cashflow_df: Optional[pd.DataFrame],
    discount_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Greenwald's Earnings Power Value: what the business is worth if it never grows.

    EPV = normalized after-tax operating earnings / cost of capital, plus net cash.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )
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
                if ebit is not None:
                    ebit_history.append(ebit)

        if not ebit_history:
            return {"error": "No operating income history for earnings power"}

        normalized_ebit = investa_core.np_median([float(e) for e in ebit_history])
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

        cash = ticker_info.get("totalCash")
        if cash is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            cash = _get_statement_value(
                balance_sheet_df, "Total Cash", balance_sheet_df.columns[0]
            )
        cash = cash or 0

        debt = ticker_info.get("totalDebt")
        if debt is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            debt = _get_statement_value(
                balance_sheet_df, "Total Debt", balance_sheet_df.columns[0]
            )
        debt = debt or 0

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
            "when_to_use": "Conservative valuation of normalized sustainable operating earnings in perpetuity assuming zero future growth.",
            "best_suited_for": "Conservative valuation of normalized sustainable operating earnings in perpetuity assuming zero future growth.",
            "key_limitation": "Strictly a no-growth baseline floor; gives zero credit to value-accretive capital reinvestment or growth opportunities.",
            "key_caveats": "Strictly a no-growth baseline floor; gives zero credit to value-accretive capital reinvestment or growth opportunities.",
            "parameters": {
                "normalized_ebit": normalized_ebit,
                "ebit_years": len(ebit_history),
                "tax_rate": tax_rate,
                "discount_rate": discount_rate,
                "nopat": nopat,
                "net_cash": cash - debt,
                "shares": shares,
                "note": "Assumes zero growth; value of current earning power only",
            },
        }
    except Exception as e:
        return {"error": f"EPV calculation failed: {str(e)}"}


def calculate_intrinsic_value_dcfo(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    discount_rate: Optional[float] = None,
    growth_rate: Optional[float] = None,
    projection_years: int = 10,
    terminal_growth_rate: float = 0.02,
    base_cfo: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Discounted Cash from Operations (D-CFO).
    Best Suited For: Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).
    Key Caveats: Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )
        if not shares or shares <= 0:
            return {"error": "Missing shares outstanding"}

        if base_cfo is None:
            if cashflow_df is not None and not cashflow_df.empty:
                base_cfo = _get_statement_value(
                    cashflow_df, "Operating Cash Flow", cashflow_df.columns[0]
                )
            if base_cfo is None:
                base_cfo = ticker_info.get("operatingCashflow")

        if base_cfo is None or base_cfo <= 0:
            return {"error": "Negative or missing Operating Cash Flow"}

        cfo_per_share = base_cfo / shares

        # Growth rate estimation
        growth_method = "caller-supplied growth"
        if growth_rate is None:
            growth_res = blended_growth_estimate(
                cashflow_df if cashflow_df is not None else financials_df,
                ticker_info=ticker_info,
                item_name="Operating Cash Flow",
            )
            raw_growth = growth_res["growth"]
            growth_method = growth_res["method"]
            shrunk = 0.40 * raw_growth + 0.60 * 0.04
            applied_growth = float(np.clip(shrunk, -0.05, 0.20))
        else:
            applied_growth = float(np.clip(growth_rate, -0.05, 0.20))

        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = wacc_res.get("wacc", 0.09)
        discount_rate = float(
            np.clip(discount_rate, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)
        )

        # Project CFO with mid-year discounting, Gordon terminal value
        sum_pv_cfos, _last_cfo, pv_terminal = _scalar_projection(
            cfo_per_share,
            applied_growth,
            discount_rate,
            projection_years,
            terminal_growth_rate,
        )

        cash = ticker_info.get("totalCash")
        if cash is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            cash = _get_statement_value(
                balance_sheet_df, "Total Cash", balance_sheet_df.columns[0]
            )
        cash = cash or 0

        debt = ticker_info.get("totalDebt")
        if debt is None and balance_sheet_df is not None and not balance_sheet_df.empty:
            debt = _get_statement_value(
                balance_sheet_df, "Total Debt", balance_sheet_df.columns[0]
            )
        debt = debt or 0

        net_cash_per_share = (cash - debt) / shares
        intrinsic_value = sum_pv_cfos + pv_terminal + net_cash_per_share

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Discounted CFO produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Discounted Cash from Operations",
            "when_to_use": "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).",
            "best_suited_for": "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).",
            "key_limitation": "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.",
            "key_caveats": "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.",
            "parameters": {
                "base_cfo": base_cfo,
                "cfo_per_share": cfo_per_share,
                "growth_rate": growth_rate
                if growth_rate is not None
                else applied_growth,
                "applied_growth": applied_growth,
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": projection_years,
                "net_cash_per_share": net_cash_per_share,
                "growth_method": growth_method,
                "mid_year_discounting": True,
            },
        }
    except Exception as e:
        return {"error": f"Discounted CFO calculation failed: {str(e)}"}


def calculate_intrinsic_value_dni(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    discount_rate: Optional[float] = None,
    growth_rate: Optional[float] = None,
    projection_years: int = 10,
    terminal_growth_rate: float = 0.02,
    base_eps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Discounted Net Income (D-NI).
    Best Suited For: Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.
    Key Caveats: Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.
    """
    try:
        if base_eps is None:
            base_eps = ticker_info.get("trailingEps")
        if base_eps is None and financials_df is not None and not financials_df.empty:
            base_eps = _get_statement_value(
                financials_df, "Diluted EPS", financials_df.columns[0]
            )

        if base_eps is None or base_eps <= 0:
            return {"error": "Negative or missing Net Income / EPS"}

        growth_method = "caller-supplied earnings growth"
        if growth_rate is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            raw_growth = growth_res["growth"]
            growth_method = growth_res["method"]
            shrunk = 0.40 * raw_growth + 0.60 * 0.04
            applied_growth = float(np.clip(shrunk, -0.05, 0.20))
        else:
            applied_growth = float(np.clip(growth_rate, -0.05, 0.20))

        if discount_rate is None:
            wacc_res = calculate_wacc(ticker_info, financials_df, balance_sheet_df)
            discount_rate = (
                wacc_res.get("cost_of_equity") or wacc_res.get("wacc") or 0.09
            )
        discount_rate = float(
            np.clip(discount_rate, MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE)
        )

        sum_pv_eps, _last_eps, pv_terminal = _scalar_projection(
            base_eps,
            applied_growth,
            discount_rate,
            projection_years,
            terminal_growth_rate,
        )

        intrinsic_value = sum_pv_eps + pv_terminal

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Discounted Net Income produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Discounted Net Income",
            "when_to_use": "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.",
            "best_suited_for": "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.",
            "key_limitation": "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.",
            "key_caveats": "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.",
            "parameters": {
                "base_eps": base_eps,
                "growth_rate": growth_rate
                if growth_rate is not None
                else applied_growth,
                "applied_growth": applied_growth,
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": projection_years,
                "growth_method": growth_method,
                "mid_year_discounting": True,
            },
        }
    except Exception as e:
        return {"error": f"Discounted Net Income calculation failed: {str(e)}"}


def calculate_intrinsic_value_mean_pe(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    mean_pe: Optional[float] = None,
    eps: Optional[float] = None,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mean P/E Ratio Valuation.
    Best Suited For: Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.
    Key Caveats: Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.

    The multiple is the median P/E this company has actually traded at, one
    observation per fiscal year (`history`, from `historical_mean_multiples`).
    It used to fall back to `trailingPE`, which is price/EPS — multiplying it
    back by EPS returned the current price to the cent, for 60% of the universe.
    """
    try:
        if eps is None:
            eps = ticker_info.get("trailingEps")
        if eps is None and financials_df is not None and not financials_df.empty:
            eps = _get_statement_value(
                financials_df, "Diluted EPS", financials_df.columns[0]
            )

        if eps is None or eps <= 0:
            return {"error": "Negative or missing EPS"}

        pe_source = "caller-supplied"
        if mean_pe is None:
            if not history or not history.get("value"):
                return {
                    "error": (
                        "No traded P/E history to average — a fair multiple must "
                        "come from what this company has actually changed hands at"
                    ),
                    "diagnostics": {"eps": eps},
                }
            mean_pe = float(history["value"])
            pe_source = (
                f"Median traded P/E {mean_pe:.1f}x "
                f"({history['n']} years, {history['span']})"
            )

        applied_pe = float(np.clip(mean_pe, 5.0, 45.0))
        intrinsic_value = eps * applied_pe

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Mean P/E produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Mean P/E Ratio",
            "when_to_use": "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
            "best_suited_for": "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
            "key_limitation": "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.",
            "key_caveats": "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.",
            "parameters": {
                "eps": eps,
                "mean_pe": mean_pe,
                "applied_pe": applied_pe,
                "pe_source": pe_source,
                "multiple_observations": (history or {}).get("n"),
                "multiple_span": (history or {}).get("span"),
                "multiple_p25": (history or {}).get("low"),
                "multiple_p75": (history or {}).get("high"),
            },
        }
    except Exception as e:
        return {"error": f"Mean P/E calculation failed: {str(e)}"}


def calculate_intrinsic_value_peg(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    growth_rate: Optional[float] = None,
    eps: Optional[float] = None,
    target_peg: float = 1.0,
    dividend_yield_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    PEG Ratio Fair Value.
    Best Suited For: Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.
    Key Caveats: Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.
    """
    try:
        if eps is None:
            eps = ticker_info.get("trailingEps")
        if eps is None and financials_df is not None and not financials_df.empty:
            eps = _get_statement_value(
                financials_df, "Diluted EPS", financials_df.columns[0]
            )

        if eps is None or eps <= 0:
            return {"error": "Negative or missing EPS"}

        growth_method = "caller-supplied growth"
        if growth_rate is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Net Income"
            )
            growth_rate = growth_res["growth"] * 100.0
            growth_method = growth_res["method"]

        if dividend_yield_pct is None:
            div_rate = ticker_info.get("dividendRate") or ticker_info.get(
                "trailingAnnualDividendRate"
            )
            curr_p = ticker_info.get("currentPrice") or ticker_info.get(
                "regularMarketPrice"
            )
            if div_rate and curr_p and curr_p > 0:
                dividend_yield_pct = (div_rate / curr_p) * 100.0
            elif ticker_info.get("dividendYield"):
                dividend_yield_pct = ticker_info["dividendYield"] * 100.0
            else:
                dividend_yield_pct = 0.0

        applied_growth = float(np.clip(growth_rate, 3.0, 35.0))
        target_peg = float(np.clip(target_peg, 0.5, 2.0))
        fair_pe_multiplier = float(
            np.clip(target_peg * (applied_growth + dividend_yield_pct), 5.0, 35.0)
        )
        intrinsic_value = eps * fair_pe_multiplier

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "PEG calculation produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "PEG Ratio Fair Value",
            "when_to_use": "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.",
            "best_suited_for": "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.",
            "key_limitation": "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.",
            "key_caveats": "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.",
            "parameters": {
                "eps": eps,
                "growth_rate_pct": growth_rate,
                "applied_growth_pct": applied_growth,
                "dividend_yield_pct": dividend_yield_pct,
                "target_peg": target_peg,
                "fair_pe_multiplier": fair_pe_multiplier,
                "growth_method": growth_method,
            },
        }
    except Exception as e:
        return {"error": f"PEG calculation failed: {str(e)}"}


def calculate_intrinsic_value_mean_pb(
    ticker_info: Dict[str, Any],
    balance_sheet_df: Optional[pd.DataFrame] = None,
    book_value_per_share: Optional[float] = None,
    mean_pb: Optional[float] = None,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mean P/B Ratio Valuation.
    Best Suited For: Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.
    Key Caveats: Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )

        if book_value_per_share is None:
            book_value_per_share = ticker_info.get("bookValue")
            if (
                (book_value_per_share is None or book_value_per_share <= 0)
                and balance_sheet_df is not None
                and not balance_sheet_df.empty
                and shares
                and shares > 0
            ):
                total_equity = _get_statement_value(
                    balance_sheet_df,
                    "Total Stockholder Equity",
                    balance_sheet_df.columns[0],
                )
                if total_equity and total_equity > 0:
                    book_value_per_share = total_equity / shares

        if book_value_per_share is None or book_value_per_share <= 0:
            return {"error": "Negative or missing Book Value per Share"}

        sector = (ticker_info.get("sector") or "").lower()
        pb_source = "caller-supplied"

        if mean_pb is None:
            # The company's own traded record first. The sector benchmarks that
            # used to come first (1.30x banks, 1.05x REIT NAV) are a claim about
            # the average bank, not about this one; they survive only as a last
            # resort where that average is a real domain prior. The `priceToBook`
            # fallback is gone outright — book value times price/book is price.
            if history and history.get("value"):
                mean_pb = float(history["value"])
                pb_source = (
                    f"Median traded P/B {mean_pb:.2f}x "
                    f"({history['n']} years, {history['span']})"
                )
            elif "bank" in sector or "financial" in sector:
                mean_pb = 1.30  # Standard 1.2 - 1.4x banking benchmark
                pb_source = "Banking benchmark (1.30x) — no traded history"
            elif "real estate" in sector or "reit" in sector:
                mean_pb = 1.05  # Price/NAV benchmark
                pb_source = "REIT NAV benchmark (1.05x) — no traded history"
            else:
                return {
                    "error": (
                        "No traded P/B history to average — a fair multiple must "
                        "come from what this company has actually changed hands at"
                    ),
                    "diagnostics": {"book_value_per_share": book_value_per_share},
                }

        applied_pb = float(np.clip(mean_pb, 0.4, 8.0))
        intrinsic_value = book_value_per_share * applied_pb

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Mean P/B produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Mean P/B Ratio",
            "when_to_use": "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
            "best_suited_for": "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
            "key_limitation": "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.",
            "key_caveats": "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.",
            "parameters": {
                "book_value_per_share": book_value_per_share,
                "mean_pb": mean_pb,
                "applied_pb": applied_pb,
                "pb_source": pb_source,
                "sector": ticker_info.get("sector"),
                "multiple_observations": (history or {}).get("n"),
                "multiple_span": (history or {}).get("span"),
                "multiple_p25": (history or {}).get("low"),
                "multiple_p75": (history or {}).get("high"),
            },
        }
    except Exception as e:
        return {"error": f"Mean P/B calculation failed: {str(e)}"}


def calculate_intrinsic_value_mean_ps(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    sales_per_share: Optional[float] = None,
    mean_ps: Optional[float] = None,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mean P/S Ratio Valuation.
    Best Suited For: Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.
    Key Caveats: Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )

        if sales_per_share is None:
            revenue = ticker_info.get("totalRevenue")
            if (
                revenue is None
                and financials_df is not None
                and not financials_df.empty
            ):
                revenue = _get_statement_value(
                    financials_df, "Total Revenue", financials_df.columns[0]
                )
            if revenue and shares and shares > 0:
                sales_per_share = revenue / shares

        if sales_per_share is None or sales_per_share <= 0:
            return {"error": "Negative or missing Revenue / Sales per share"}

        ps_source = "caller-supplied"
        if mean_ps is None:
            if not history or not history.get("value"):
                return {
                    "error": (
                        "No traded P/S history to average — a fair multiple must "
                        "come from what this company has actually changed hands at"
                    ),
                    "diagnostics": {"sales_per_share": sales_per_share},
                }
            mean_ps = float(history["value"])
            ps_source = (
                f"Median traded P/S {mean_ps:.2f}x "
                f"({history['n']} years, {history['span']})"
            )

        applied_ps = float(np.clip(mean_ps, 0.4, 15.0))
        intrinsic_value = sales_per_share * applied_ps

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "Mean P/S produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Mean P/S Ratio",
            "when_to_use": "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
            "best_suited_for": "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
            "key_limitation": "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.",
            "key_caveats": "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.",
            "parameters": {
                "sales_per_share": sales_per_share,
                "mean_ps": mean_ps,
                "applied_ps": applied_ps,
                "ps_source": ps_source,
                "multiple_observations": (history or {}).get("n"),
                "multiple_span": (history or {}).get("span"),
                "multiple_p25": (history or {}).get("low"),
                "multiple_p75": (history or {}).get("high"),
            },
        }
    except Exception as e:
        return {"error": f"Mean P/S calculation failed: {str(e)}"}


def calculate_intrinsic_value_psg(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    sales_per_share: Optional[float] = None,
    revenue_growth_pct: Optional[float] = None,
    target_psg: float = 1.0,
) -> Dict[str, Any]:
    """
    Price-to-Sales Growth Ratio (PSG) Valuation.
    Best Suited For: High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.
    Key Caveats: Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.
    """
    try:
        shares = ticker_info.get("sharesOutstanding")
        if not shares and balance_sheet_df is not None and not balance_sheet_df.empty:
            shares = _get_statement_value(
                balance_sheet_df, "Ordinary Shares Number", balance_sheet_df.columns[0]
            )

        if sales_per_share is None:
            revenue = ticker_info.get("totalRevenue")
            if (
                revenue is None
                and financials_df is not None
                and not financials_df.empty
            ):
                revenue = _get_statement_value(
                    financials_df, "Total Revenue", financials_df.columns[0]
                )
            if revenue and shares and shares > 0:
                sales_per_share = revenue / shares

        if sales_per_share is None or sales_per_share <= 0:
            return {"error": "Negative or missing Revenue / Sales per share"}

        growth_method = "caller-supplied growth"
        if revenue_growth_pct is None:
            growth_res = blended_growth_estimate(
                financials_df, ticker_info=ticker_info, item_name="Total Revenue"
            )
            revenue_growth_pct = growth_res["growth"] * 100.0
            growth_method = growth_res["method"]

        gross_margin = ticker_info.get("grossMargins")
        if (
            gross_margin is None
            and financials_df is not None
            and not financials_df.empty
        ):
            rev = _get_statement_value(
                financials_df, "Total Revenue", financials_df.columns[0]
            )
            gp = _get_statement_value(
                financials_df, "Gross Profit", financials_df.columns[0]
            )
            if rev and gp and rev > 0:
                gross_margin = gp / rev
        gross_margin = float(
            np.clip(gross_margin if gross_margin is not None else 0.50, 0.20, 0.95)
        )

        applied_growth = float(np.clip(revenue_growth_pct, 5.0, 45.0))
        target_psg = float(np.clip(target_psg, 0.5, 2.0))

        # Fair P/S multiplier scales with revenue growth rate scaled by gross margin quality
        fair_ps_multiplier = float(
            np.clip(
                (applied_growth / 10.0) * gross_margin * target_psg * 1.5, 0.5, 12.0
            )
        )
        intrinsic_value = sales_per_share * fair_ps_multiplier

        if not np.isfinite(intrinsic_value) or intrinsic_value <= 0:
            return {"error": "PSG calculation produced non-positive value"}

        return {
            "intrinsic_value": intrinsic_value,
            "model": "Price-to-Sales Growth (PSG)",
            "when_to_use": "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.",
            "best_suited_for": "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.",
            "key_limitation": "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.",
            "key_caveats": "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.",
            "parameters": {
                "sales_per_share": sales_per_share,
                "revenue_growth_pct": revenue_growth_pct,
                "applied_growth_pct": applied_growth,
                "gross_margin_pct": gross_margin * 100.0,
                "target_psg": target_psg,
                "fair_ps_multiplier": fair_ps_multiplier,
                "growth_method": growth_method,
            },
        }
    except Exception as e:
        return {"error": f"PSG calculation failed: {str(e)}"}


def recommend_best_valuation_method(
    ticker_info: Dict[str, Any],
    models: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    is_bank_or_financial: bool = False,
) -> Dict[str, Any]:
    """
    Selects the best-fit valuation method according to the 'Choosing the Right Valuation Method' framework:
    1. Financials/Banks/REITs -> Mean P/B (for banks/REITs) or Discounted Net Income (D-NI).
    2. High-Growth Unprofitable -> Price-to-Sales Growth (PSG) or Mean P/S.
    3. Steady Cash Generator -> Discounted Free Cash Flow (DCF - Primary Method).
    4. Lumpy CapEx but Steady Operations -> Discounted Cash from Operations (D-CFO).
    5. Consistent Earnings Growth -> PEG Ratio or Mean P/E.
    """
    sector = (ticker_info.get("sector") or "").lower()
    industry = (ticker_info.get("industry") or "").lower()

    if (
        "bank" in sector
        or "financial" in sector
        or "bank" in industry
        or "insurance" in industry
        or "real estate" in sector
    ):
        is_bank_or_financial = True

    eps = ticker_info.get("trailingEps") or 0.0
    fcf = ticker_info.get("freeCashflow") or 0.0
    cfo = ticker_info.get("operatingCashflow") or 0.0
    rev_growth = ticker_info.get("revenueGrowth") or 0.0
    div_yield = ticker_info.get("dividendYield") or 0.0

    # 1. Banks, REITs, Financials
    if is_bank_or_financial:
        if "bank" in sector or "real estate" in sector:
            if models.get("mean_pb", {}).get("intrinsic_value") is not None:
                return {
                    "method_key": "mean_pb",
                    "name": "Mean P/B Ratio",
                    "when_to_use": "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
                    "best_suited_for": "Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
                    "key_limitation": "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.",
                    "key_caveats": "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.",
                    "rationale": "Financial institutions and property assets are primarily asset-backed and valued relative to book value and marked-to-market equity.",
                    "intrinsic_value": models["mean_pb"]["intrinsic_value"],
                }
        if models.get("dni", {}).get("intrinsic_value") is not None:
            return {
                "method_key": "dni",
                "name": "Discounted Net Income",
                "when_to_use": "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.",
                "best_suited_for": "Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.",
                "key_limitation": "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.",
                "key_caveats": "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.",
                "rationale": "Financial firms do not have traditional industrial CapEx; discounting Net Income at Cost of Equity is the canonical approach.",
                "intrinsic_value": models["dni"]["intrinsic_value"],
            }
        if (
            models.get("ddm", {}).get("intrinsic_value") is not None
            and div_yield > 0.02
        ):
            return {
                "method_key": "ddm",
                "name": "Dividend Discount Model",
                "when_to_use": "Mature dividend payers and utilities with long track records of consistent dividend growth and sustainable payout ratios (<100%).",
                "best_suited_for": "Mature dividend payers and utilities with long track records of consistent dividend growth and sustainable payout ratios (<100%).",
                "key_limitation": "Only reflects value returned as direct dividends; entirely unsuited for non-dividend payers and ignores share repurchases or cash retained.",
                "key_caveats": "Only reflects value returned as direct dividends; entirely unsuited for non-dividend payers and ignores share repurchases or cash retained.",
                "rationale": "High dividend payout yields direct cash return to shareholders.",
                "intrinsic_value": models["ddm"]["intrinsic_value"],
            }

    # 2. Speculative / Unprofitable High Growth
    if eps <= 0 and rev_growth >= 0.12:
        if models.get("psg", {}).get("intrinsic_value") is not None:
            return {
                "method_key": "psg",
                "name": "Price-to-Sales Growth (PSG)",
                "when_to_use": "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.",
                "best_suited_for": "High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.",
                "key_limitation": "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.",
                "key_caveats": "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.",
                "rationale": "Unprofitable hyper-growth companies require growth-relative revenue multiples while unit economics scale.",
                "intrinsic_value": models["psg"]["intrinsic_value"],
            }
        if models.get("mean_ps", {}).get("intrinsic_value") is not None:
            return {
                "method_key": "mean_ps",
                "name": "Mean P/S Ratio",
                "when_to_use": "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
                "best_suited_for": "Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
                "key_limitation": "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.",
                "key_caveats": "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.",
                "rationale": "Historical sales multiple baseline for companies without consistent earnings.",
                "intrinsic_value": models["mean_ps"]["intrinsic_value"],
            }

    # 3. Predictable Free Cash Flow Generator -> DCF (Primary Method)
    if models.get("dcf", {}).get("intrinsic_value") is not None and fcf > 0:
        return {
            "method_key": "dcf",
            "name": "Discounted Free Cash Flow (Primary)",
            "when_to_use": "Cash-generative companies with steady, predictable Free Cash Flow (Operating Cash Flow minus Capital Expenditures).",
            "best_suited_for": "Cash-generative companies with steady, predictable Free Cash Flow (Operating Cash Flow minus Capital Expenditures).",
            "key_limitation": "Highly sensitive to growth and discount rate (WACC) inputs; unsuitable for cyclical, negative-FCF, or lumpy CapEx businesses.",
            "key_caveats": "Highly sensitive to growth and discount rate (WACC) inputs; unsuitable for cyclical, negative-FCF, or lumpy CapEx businesses.",
            "rationale": "Strong, predictable Free Cash Flow makes DCF the gold standard valuation model.",
            "intrinsic_value": models["dcf"]["intrinsic_value"],
        }

    # 4. Steady Operating Cash Flow but Lumpy CapEx -> D-CFO
    if models.get("dcfo", {}).get("intrinsic_value") is not None and cfo > 0:
        return {
            "method_key": "dcfo",
            "name": "Discounted Cash from Operations",
            "when_to_use": "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).",
            "best_suited_for": "Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).",
            "key_limitation": "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.",
            "key_caveats": "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.",
            "rationale": "Operating cash flow is durable while capital expenditure fluctuates through reinvestment cycles.",
            "intrinsic_value": models["dcfo"]["intrinsic_value"],
        }

    # 5. Steady Earnings -> PEG or Mean P/E
    if models.get("peg", {}).get("intrinsic_value") is not None and eps > 0:
        return {
            "method_key": "peg",
            "name": "PEG Ratio Fair Value",
            "when_to_use": "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.",
            "best_suited_for": "Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.",
            "key_limitation": "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.",
            "key_caveats": "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.",
            "rationale": "Growth-adjusted earnings multiple benchmark matching earnings expansion with valuation.",
            "intrinsic_value": models["peg"]["intrinsic_value"],
        }

    if models.get("mean_pe", {}).get("intrinsic_value") is not None and eps > 0:
        return {
            "method_key": "mean_pe",
            "name": "Mean P/E Ratio",
            "when_to_use": "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
            "best_suited_for": "Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
            "key_limitation": "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.",
            "key_caveats": "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.",
            "rationale": "Mature, profitable company with established earnings multiple baseline.",
            "intrinsic_value": models["mean_pe"]["intrinsic_value"],
        }

    # Fallback to Graham or whatever model succeeded
    for k in ("graham", "epv", "lynch", "mean_pb", "mean_ps"):
        if models.get(k, {}).get("intrinsic_value") is not None:
            return {
                "method_key": k,
                "name": models[k].get("model", k.upper()),
                "when_to_use": "Specialized fallback valuation model applicable to available reporting.",
                "best_suited_for": "Specialized fallback valuation model applicable to available reporting.",
                "key_limitation": "May not account for all future growth phases or capital structure nuances.",
                "key_caveats": "May not account for all future growth phases or capital structure nuances.",
                "rationale": "Alternative baseline model applicable to available reporting.",
                "intrinsic_value": models[k]["intrinsic_value"],
            }

    return {
        "method_key": "none",
        "name": "No Defensible Model",
        "when_to_use": "N/A",
        "best_suited_for": "N/A",
        "key_limitation": "N/A",
        "key_caveats": "N/A",
        "rationale": "Fundamental data is insufficient to compute a reliable intrinsic value.",
        "intrinsic_value": None,
    }


def _stable_mc_seed(ticker_info: Dict[str, Any]) -> int:
    """A seed that depends on the company, not on when it was asked about."""
    key = str(
        ticker_info.get("symbol")
        or ticker_info.get("shortName")
        or ticker_info.get("longName")
        or "investa"
    )
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "big")


@contextlib.contextmanager
def _deterministic_draws(ticker_info: Dict[str, Any]):
    """Seed the global RNG for one company's simulations, then put it back."""
    state = np.random.get_state()
    np.random.seed(_stable_mc_seed(ticker_info))
    try:
        yield
    finally:
        np.random.set_state(state)


def _blend_profile(ticker_info: Dict[str, Any]) -> str:
    """Which family of models can describe this business at all.

    Two groups the old weight table got wrong by lumping them in with banks:

    - A *utility* has real capex and real free cash flow, so the DCF describes
      it. Its high payout is picked up by the dividend gate on its own.
    - A *property developer* is not a REIT. It sells buildings rather than
      distributing rent, so it belongs with operating companies; only the trusts
      themselves get the REIT treatment.
    """
    sector = (ticker_info.get("sector") or "").lower()
    industry = (ticker_info.get("industry") or "").lower()
    if "reit" in industry or "reit" in sector:
        return "reit"
    if any(k in f"{sector} {industry}" for k in ("financial", "bank", "insurance")):
        return "financial"
    return "operating"


def _model_reliability(model_res: Optional[Dict[str, Any]]) -> float:
    """Confidence in one model's own number, from its Monte Carlo dispersion.

    Returns 1.0 when the model ran no simulation — absence of evidence about
    uncertainty is not evidence of uncertainty, and penalising it would quietly
    prefer whichever models happen to simulate.
    """
    mc = (model_res or {}).get("mc") or {}
    bear, base, bull = mc.get("bear"), mc.get("base"), mc.get("bull")
    try:
        if bear is None or base is None or bull is None:
            return 1.0
        bear, base, bull = float(bear), float(base), float(bull)
    except (TypeError, ValueError):
        return 1.0
    return investa_core.model_reliability(
        bear, base, bull, RELIABILITY_CV_REF, RELIABILITY_FLOOR
    )


def _ddm_describes_the_company(ticker_info: Dict[str, Any], profile: str) -> bool:
    """True when the dividend is substantially the whole distributable cash."""
    if profile == "reit":
        return True
    payout = ticker_info.get("payoutRatio")
    try:
        payout = float(payout)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(payout) and payout >= DDM_MIN_PAYOUT_FOR_BLEND)


def blend_intrinsic_values(
    models: Dict[str, Any], ticker_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine the model estimates into one fair value, with its confidence.

    Weight is `prior x reliability`: the prior says which models can describe
    this kind of business, the reliability says how tightly each one pinned its
    own answer for this company. A weighted *arithmetic* mean aggregates them —
    a geometric mean and a Huber down-weight of outlying models were both
    measured on the cached universe and both made calibration and rank agreement
    with earnings yield slightly worse, so neither is here.
    """
    profile = _blend_profile(ticker_info)
    priors = {
        "financial": MODEL_BLEND_PRIORS_FINANCIAL,
        "reit": MODEL_BLEND_PRIORS_REIT,
    }.get(profile, MODEL_BLEND_PRIORS_OPERATING)

    contributions: List[Dict[str, Any]] = []
    excluded: Dict[str, str] = {}

    for key, prior in priors.items():
        model_res = models.get(key) or {}
        iv = model_res.get("intrinsic_value")
        if iv is None or not np.isfinite(iv) or iv <= 0:
            continue
        if key == "ddm" and not _ddm_describes_the_company(ticker_info, profile):
            excluded[key] = (
                "Payout ratio below "
                f"{DDM_MIN_PAYOUT_FOR_BLEND:.0%} — the dividend is not the whole "
                "of distributable cash, so DDM values the dividend rather than "
                "the company"
            )
            continue
        reliability = _model_reliability(model_res)
        contributions.append(
            {
                "key": key,
                "value": float(iv),
                "prior": float(prior),
                "reliability": reliability,
                "weight": float(prior) * reliability,
            }
        )

    if not contributions:
        return {
            "value": None,
            "profile": profile,
            "weights": {},
            "contributions": [],
            "spread_pct": None,
            "confidence": 0.0,
            "excluded": excluded,
        }

    # Weighted arithmetic mean, max-min spread, and a confidence that is
    # continuous, not a cliff: how tightly the models pinned their own answers,
    # how far apart they landed, and how many of them there were. Computed in
    # Rust (investa_core.blend).
    value, norm_weights, spread_pct, confidence = investa_core.blend(
        [c["value"] for c in contributions],
        [c["prior"] for c in contributions],
        [c["reliability"] for c in contributions],
    )
    weights = {c["key"]: w for c, w in zip(contributions, norm_weights)}

    return {
        "value": float(value),
        "profile": profile,
        "weights": weights,
        "contributions": contributions,
        "spread_pct": spread_pct,
        "confidence": confidence,
        "excluded": excluded,
    }


def get_comprehensive_intrinsic_value(
    ticker_info: Dict[str, Any],
    financials_df: Optional[pd.DataFrame] = None,
    balance_sheet_df: Optional[pd.DataFrame] = None,
    cashflow_df: Optional[pd.DataFrame] = None,
    overrides: Optional[Dict[str, Any]] = None,
    iterations: int = 10000,
    prices_df: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Consolidates multiple intrinsic value models into a single advice object.
    Supports DCF, D-CFO, D-NI, Mean P/E, PEG, Mean P/B, Mean P/S, PSG, Graham, DDM, EPV, and Peter Lynch fair value.
    """
    # 0. Enrich data with statement fallbacks to maximize coverage
    ticker_info = _enrich_ticker_info(
        ticker_info, financials_df, balance_sheet_df, cashflow_df
    )

    overrides = overrides or {}

    # What the market has paid for this company historically, one observation
    # per fiscal year. Empty without price history, and the three multiple
    # models refuse rather than reaching for today's multiple.
    traded_multiples = historical_mean_multiples(
        financials_df, balance_sheet_df, cashflow_df, prices_df
    )

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

    # Extract DDM overrides
    ddm_discount = overrides.get("ddm_discount_rate")
    ddm_growth = overrides.get("ddm_growth_rate")
    ddm_terminal = overrides.get("ddm_terminal_growth", 0.02)
    ddm_projection = int(overrides.get("ddm_projection_years", 10))
    ddm_dividend = overrides.get("ddm_base_dividend")

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

    dcfo_res = calculate_intrinsic_value_dcfo(
        ticker_info,
        financials_df,
        balance_sheet_df,
        cashflow_df,
        discount_rate=overrides.get("dcfo_discount_rate", dcf_discount),
        growth_rate=overrides.get("dcfo_growth_rate", dcf_growth),
        projection_years=dcf_projection,
        terminal_growth_rate=dcf_terminal,
        base_cfo=overrides.get("dcfo_base_cfo"),
    )

    dni_res = calculate_intrinsic_value_dni(
        ticker_info,
        financials_df,
        balance_sheet_df,
        discount_rate=overrides.get("dni_discount_rate", ddm_discount),
        growth_rate=overrides.get("dni_growth_rate", graham_growth),
        projection_years=dcf_projection,
        terminal_growth_rate=dcf_terminal,
        base_eps=overrides.get("dni_base_eps", graham_eps),
    )

    mean_pe_res = calculate_intrinsic_value_mean_pe(
        ticker_info,
        financials_df,
        eps=overrides.get("mean_pe_eps", graham_eps),
        mean_pe=overrides.get("mean_pe_multiple"),
        history=traded_multiples.get("pe"),
    )

    peg_res = calculate_intrinsic_value_peg(
        ticker_info,
        financials_df,
        growth_rate=overrides.get("peg_growth_rate", graham_growth),
        eps=overrides.get("peg_eps", graham_eps),
        target_peg=overrides.get("peg_target", 1.0),
    )

    mean_pb_res = calculate_intrinsic_value_mean_pb(
        ticker_info,
        balance_sheet_df,
        book_value_per_share=overrides.get("mean_pb_bvps"),
        mean_pb=overrides.get("mean_pb_multiple"),
        history=traded_multiples.get("pb"),
    )

    mean_ps_res = calculate_intrinsic_value_mean_ps(
        ticker_info,
        financials_df,
        balance_sheet_df,
        sales_per_share=overrides.get("mean_ps_sps"),
        mean_ps=overrides.get("mean_ps_multiple"),
        history=traded_multiples.get("ps"),
    )

    psg_res = calculate_intrinsic_value_psg(
        ticker_info,
        financials_df,
        balance_sheet_df,
        sales_per_share=overrides.get("psg_sps"),
        revenue_growth_pct=overrides.get("psg_revenue_growth"),
        target_psg=overrides.get("psg_target", 1.0),
    )

    graham_res = calculate_intrinsic_value_graham(
        ticker_info,
        financials_df,
        balance_sheet_df,
        growth_rate=graham_growth,
        eps=graham_eps,
        bond_yield=graham_bond_yield,
    )

    ddm_res = calculate_intrinsic_value_ddm(
        ticker_info,
        financials_df,
        cashflow_df,
        balance_sheet_df,
        discount_rate=ddm_discount,
        growth_rate=ddm_growth,
        projection_years=ddm_projection,
        terminal_growth_rate=ddm_terminal,
        base_dividend=ddm_dividend,
    )

    epv_res = calculate_earnings_power_value(
        ticker_info,
        financials_df,
        balance_sheet_df,
        cashflow_df,
        discount_rate=overrides.get("epv_discount_rate", dcf_discount),
    )

    lynch_res = calculate_intrinsic_value_lynch(
        ticker_info,
        financials_df,
        growth_rate=overrides.get("lynch_growth_rate", graham_growth),
        eps=overrides.get("lynch_eps", graham_eps),
    )

    # Monte Carlo, seeded per company. The draws feed the reliability weights
    # below, so unseeded sampling would make the headline fair value wobble
    # between two calls on the same company — small, but a valuation that
    # changes on refresh is not a valuation. The global RNG is restored on the
    # way out so callers keep their own stream.
    with _deterministic_draws(ticker_info):
        # Run Monte Carlo simulations for all 12 valuation models
        if "intrinsic_value" in dcf_res and "parameters" in dcf_res:
            params = dcf_res["parameters"]
            mc_growth = params.get("applied_growth", params.get("growth_rate", 0.05))
            dcf_res["mc"] = run_monte_carlo_dcf(
                ticker_info,
                params["base_fcf"],
                mc_growth,
                params["discount_rate"],
                params["projection_years"],
                params["terminal_growth_rate"],
                iterations=iterations,
            )

        if "intrinsic_value" in dcfo_res and "parameters" in dcfo_res:
            params = dcfo_res["parameters"]
            mc_growth = params.get("applied_growth", params.get("growth_rate", 0.05))
            dcfo_res["mc"] = run_monte_carlo_dcfo(
                ticker_info,
                params["base_cfo"],
                mc_growth,
                params["discount_rate"],
                params["projection_years"],
                params["terminal_growth_rate"],
                iterations=iterations,
            )

        if "intrinsic_value" in dni_res and "parameters" in dni_res:
            params = dni_res["parameters"]
            mc_growth = params.get("applied_growth", params.get("growth_rate", 0.05))
            dni_res["mc"] = run_monte_carlo_dni(
                params["base_eps"],
                mc_growth,
                params["discount_rate"],
                params["projection_years"],
                params["terminal_growth_rate"],
                iterations=iterations,
            )

        if "intrinsic_value" in mean_pe_res and "parameters" in mean_pe_res:
            params = mean_pe_res["parameters"]
            mean_pe_res["mc"] = run_monte_carlo_mean_pe(
                params["eps"],
                params["applied_pe"],
                iterations=iterations,
            )

        if "intrinsic_value" in peg_res and "parameters" in peg_res:
            params = peg_res["parameters"]
            peg_res["mc"] = run_monte_carlo_peg(
                params["eps"],
                params.get("applied_growth_pct", params.get("growth_rate_pct", 10.0)),
                params.get("dividend_yield_pct", 0.0),
                params.get("target_peg", 1.0),
                iterations=iterations,
            )

        if "intrinsic_value" in mean_pb_res and "parameters" in mean_pb_res:
            params = mean_pb_res["parameters"]
            mean_pb_res["mc"] = run_monte_carlo_mean_pb(
                params["book_value_per_share"],
                params["applied_pb"],
                iterations=iterations,
            )

        if "intrinsic_value" in mean_ps_res and "parameters" in mean_ps_res:
            params = mean_ps_res["parameters"]
            mean_ps_res["mc"] = run_monte_carlo_mean_ps(
                params["sales_per_share"],
                params["applied_ps"],
                iterations=iterations,
            )

        if "intrinsic_value" in psg_res and "parameters" in psg_res:
            params = psg_res["parameters"]
            gm = (
                params.get("gross_margin_pct", 50.0) / 100.0
                if "gross_margin_pct" in params
                else 0.50
            )
            psg_res["mc"] = run_monte_carlo_psg(
                params["sales_per_share"],
                params.get(
                    "applied_growth_pct", params.get("revenue_growth_pct", 15.0)
                ),
                gm,
                params.get("target_psg", 1.0),
                iterations=iterations,
            )

        if "intrinsic_value" in graham_res and "parameters" in graham_res:
            params = graham_res["parameters"]
            if "eps" in params:
                mc_growth_pct = params.get(
                    "applied_growth_pct", params.get("growth_rate_pct", 0)
                )
                graham_res["mc"] = run_monte_carlo_graham(
                    params["eps"],
                    mc_growth_pct,
                    params["bond_yield_proxy"],
                    iterations=iterations,
                )

        if "intrinsic_value" in ddm_res and "parameters" in ddm_res:
            params = ddm_res["parameters"]
            ddm_res["mc"] = run_monte_carlo_ddm(
                params["base_dividend"],
                params["applied_growth"],
                params["discount_rate"],
                params["projection_years"],
                params["terminal_growth_rate"],
                iterations=iterations,
            )

        if "intrinsic_value" in epv_res and "parameters" in epv_res:
            params = epv_res["parameters"]
            epv_res["mc"] = run_monte_carlo_epv(
                ticker_info,
                params["normalized_ebit"],
                params["tax_rate"],
                params["discount_rate"],
                iterations=iterations,
            )

        if "intrinsic_value" in lynch_res and "parameters" in lynch_res:
            params = lynch_res["parameters"]
            lynch_res["mc"] = run_monte_carlo_lynch(
                params["eps"],
                params.get("growth_rate_pct", 10.0),
                params.get("dividend_yield_pct", 0.0),
                iterations=iterations,
            )

    current_price = ticker_info.get("currentPrice") or ticker_info.get(
        "regularMarketPrice"
    )

    # --- ETF Valuation Logic ---
    quote_type = ticker_info.get("quoteType", "").upper()
    if quote_type == "ETF" or quote_type == "MUTUALFUND":
        nav_price = ticker_info.get("navPrice")
        if (nav_price is None or nav_price == 0) and current_price:
            nav_price = current_price

        if nav_price:
            results = {
                "current_price": current_price,
                "average_intrinsic_value": nav_price,
                "valuation_note": f"Valuation based on Net Asset Value (NAV) for {quote_type}.",
                "valuation_status": "nav",
                # NAV is reported by the fund, not estimated from models.
                "valuation_confidence": 1.0,
                "recommended_method": {
                    "method_key": "nav",
                    "name": "Net Asset Value (NAV)",
                    "when_to_use": "ETFs, CEFs, and Mutual Funds holding diversified portfolios of publicly-traded securities.",
                    "best_suited_for": "ETFs, CEFs, and Mutual Funds holding diversified portfolios of publicly-traded securities.",
                    "key_limitation": "Fund NAV equals total underlying portfolio assets minus fund liabilities, reflecting portfolio holdings rather than operating business cash flows.",
                    "key_caveats": "Fund NAV equals total underlying portfolio assets minus fund liabilities, reflecting portfolio holdings rather than operating business cash flows.",
                    "rationale": "Funds are valued by their reported Net Asset Value per share.",
                    "intrinsic_value": nav_price,
                },
                "models": {
                    "dcf": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "dcfo": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "dni": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "mean_pe": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "peg": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "mean_pb": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "mean_ps": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "psg": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "graham": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "ddm": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "epv": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                    "lynch": {"model": "N/A (ETF/Fund)", "intrinsic_value": None},
                },
            }

            if current_price:
                mos = ((nav_price - current_price) / current_price) * 100
                if mos is not None and (np.isnan(mos) or np.isinf(mos)):
                    mos = None
                results["margin_of_safety_pct"] = mos

            return results

    models_map = {
        "dcf": dcf_res,
        "dcfo": dcfo_res,
        "dni": dni_res,
        "mean_pe": mean_pe_res,
        "peg": peg_res,
        "mean_pb": mean_pb_res,
        "mean_ps": mean_ps_res,
        "psg": psg_res,
        "graham": graham_res,
        "ddm": ddm_res,
        "epv": epv_res,
        "lynch": lynch_res,
    }

    # Best-fit method recommendation
    recommended = recommend_best_valuation_method(
        ticker_info, models_map, financials_df, cashflow_df
    )

    results: Dict[str, Any] = {
        "current_price": current_price,
        "recommended_method": recommended,
        "models": models_map,
    }

    # --- Eligibility ---------------------------------------------------------
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
        results["valuation_confidence"] = 0.0
        results["valuation_note"] = ineligible
        return results

    # --- Business-aware, uncertainty-weighted blend --------------------------
    blend = blend_intrinsic_values(models_map, ticker_info)
    contributions = blend["contributions"]

    # EPV travels alongside the estimate as the no-growth floor, not inside it.
    epv_iv = epv_res.get("intrinsic_value")
    if epv_iv is not None and np.isfinite(epv_iv) and epv_iv > 0:
        results["earnings_power_floor"] = float(epv_iv)

    lynch_iv = lynch_res.get("intrinsic_value")
    if lynch_iv is not None and np.isfinite(lynch_iv) and lynch_iv > 0:
        results["lynch_fair_value"] = float(lynch_iv)

    # A DDM held out by the payout gate is still worth showing — it is what the
    # dividend stream alone is worth, which is a floor in the same sense as EPV.
    ddm_iv = ddm_res.get("intrinsic_value")
    if "ddm" in blend["excluded"] and ddm_iv is not None and np.isfinite(ddm_iv):
        results["dividend_discount_floor"] = float(ddm_iv)

    results["blend_profile"] = blend["profile"]
    if blend["excluded"]:
        results["blend_exclusions"] = blend["excluded"]

    if not contributions:
        eligible = {
            "financial": (dni_res, ddm_res),
            "reit": (dcfo_res, ddm_res),
        }.get(blend["profile"], (dcf_res, graham_res, ddm_res))
        reasons = [m.get("error") for m in eligible if m.get("error")]
        reasons += list(blend["excluded"].values())
        results["average_intrinsic_value"] = None
        results["valuation_status"] = "no_model"
        results["valuation_confidence"] = 0.0
        results["valuation_note"] = (
            "No model could value this company: " + "; ".join(dict.fromkeys(reasons))
            if reasons
            else "No model could value this company."
        )
        return results

    avg_intrinsic = blend["value"]
    results["model_weights"] = blend["weights"]
    results["model_reliability"] = {c["key"]: c["reliability"] for c in contributions}
    spread_pct = blend["spread_pct"]
    confidence = blend["confidence"]

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

    if status == "clamped":
        # The published number is no longer the one the models produced.
        confidence *= 0.5

    if spread_pct is not None and spread_pct > 100:
        detail = ", ".join(
            "{}={:.2f}".format(c["key"], c["value"]) for c in contributions
        )
        notes.append(
            f"Models disagree by {spread_pct:.0f}% of the blended value ({detail})."
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD and status == "ok":
        status = "low_confidence"
        if not notes:
            notes.append(
                f"Low confidence ({confidence:.0%}): the contributing models are "
                "individually uncertain or few in number."
            )

    if not np.isfinite(avg_intrinsic):
        avg_intrinsic = None
        status = "no_model"
        confidence = 0.0

    results["average_intrinsic_value"] = avg_intrinsic
    results["valuation_status"] = status
    results["valuation_confidence"] = round(float(confidence), 3)
    results["model_spread_pct"] = spread_pct
    if notes:
        results["valuation_note"] = " ".join(notes)

    # Probabilistic range, built from the *contributing* models at their blend
    # weights. Averaging all twelve simulations produced a band around a number
    # nothing in it had computed — it could sit off-centre from the headline or
    # fail to contain it at all.
    weighted_bear, weighted_bull, band_weight = 0.0, 0.0, 0.0
    for c in contributions:
        mc = (models_map.get(c["key"]) or {}).get("mc") or {}
        bear, bull = mc.get("bear"), mc.get("bull")
        if bear is None or bull is None:
            continue
        if not (np.isfinite(bear) and np.isfinite(bull)):
            continue
        w = blend["weights"].get(c["key"], 0.0)
        weighted_bear += bear * w
        weighted_bull += bull * w
        band_weight += w
    if band_weight > 0:
        results["range"] = {
            "bear": weighted_bear / band_weight,
            "bull": weighted_bull / band_weight,
        }

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
            prices_df=load_price_history_for_multiples(mdp, yf_symbol),
        )

        return results
    except Exception as e:
        logging.error(f"Error in get_intrinsic_value_for_symbol for {yf_symbol}: {e}")
        return {"error": str(e)}
