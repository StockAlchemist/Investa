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

import hashlib  # noqa: F401 -- read by the Rust port at call time

import pandas as pd
import investa_core
import numpy as np
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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_dcf(
        ticker_info, financials_df, balance_sheet_df, cashflow_df, discount_rate,
        growth_rate, projection_years, terminal_growth_rate, target_fcf_margin, fcf,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_graham(
        ticker_info, financials_df, balance_sheet_df, growth_rate, eps, bond_yield
    )


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_ddm(
        ticker_info, financials_df, cashflow_df, balance_sheet_df, discount_rate,
        growth_rate, projection_years, terminal_growth_rate, base_dividend,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_lynch(ticker_info, financials_df, growth_rate, eps)


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_epv(
        ticker_info, financials_df, balance_sheet_df, cashflow_df, discount_rate
    )


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_dcfo(
        ticker_info, financials_df, balance_sheet_df, cashflow_df, discount_rate,
        growth_rate, projection_years, terminal_growth_rate, base_cfo,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_dni(
        ticker_info, financials_df, balance_sheet_df, discount_rate,
        growth_rate, projection_years, terminal_growth_rate, base_eps,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_mean_pe(ticker_info, financials_df, mean_pe, eps, history)


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_peg(
        ticker_info, financials_df, growth_rate, eps, target_peg, dividend_yield_pct
    )


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_mean_pb(
        ticker_info, balance_sheet_df, book_value_per_share, mean_pb, history
    )


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_mean_ps(
        ticker_info, financials_df, balance_sheet_df, sales_per_share, mean_ps, history
    )


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
    # Ported to Rust: rust/investa_core/src/fr_models.rs
    return investa_core.fr_iv_psg(
        ticker_info, financials_df, balance_sheet_df, sales_per_share,
        revenue_growth_pct, target_psg,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_recommend_best_valuation_method(
        ticker_info, models, financials_df, cashflow_df, is_bank_or_financial
    )


def _stable_mc_seed(ticker_info: Dict[str, Any]) -> int:
    """A seed that depends on the company, not on when it was asked about."""
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_stable_mc_seed(ticker_info)


def _deterministic_draws(ticker_info: Dict[str, Any]):
    """Seed the global RNG for one company's simulations, then put it back."""
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.DeterministicDraws(ticker_info)


def _blend_profile(ticker_info: Dict[str, Any]) -> str:
    """Which family of models can describe this business at all.

    Two groups the old weight table got wrong by lumping them in with banks:

    - A *utility* has real capex and real free cash flow, so the DCF describes
      it. Its high payout is picked up by the dividend gate on its own.
    - A *property developer* is not a REIT. It sells buildings rather than
      distributing rent, so it belongs with operating companies; only the trusts
      themselves get the REIT treatment.
    """
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_blend_profile(ticker_info)


def _model_reliability(model_res: Optional[Dict[str, Any]]) -> float:
    """Confidence in one model's own number, from its Monte Carlo dispersion.

    Returns 1.0 when the model ran no simulation — absence of evidence about
    uncertainty is not evidence of uncertainty, and penalising it would quietly
    prefer whichever models happen to simulate.
    """
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_model_reliability(model_res)


def _ddm_describes_the_company(ticker_info: Dict[str, Any], profile: str) -> bool:
    """True when the dividend is substantially the whole distributable cash."""
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_ddm_describes_the_company(ticker_info, profile)


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
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_blend_intrinsic_values(models, ticker_info)


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
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_comprehensive_intrinsic_value(
        ticker_info, financials_df, balance_sheet_df, cashflow_df,
        overrides, iterations, prices_df,
    )  # fmt: skip


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
    # Ported to Rust: rust/investa_core/src/fr_blend.rs
    return investa_core.fr_intrinsic_value_for_symbol(
        symbol, mdp, config_manager, iterations, force_refresh,
        prefetched_financials, prefetched_balance_sheet, prefetched_cashflow,
    )  # fmt: skip
