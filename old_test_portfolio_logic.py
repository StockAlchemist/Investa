# --- START OF FILE test_portfolio_logic.py ---

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from io import StringIO
from collections import defaultdict

# --- Functions/Constants to test ---
# Note: Adjust the import path if your test file is not in the same directory
from portfolio_logic import (
    calculate_npv,
    calculate_irr,
    get_conversion_rate,
    get_historical_price,  # Will need mock data
    load_and_clean_transactions,
    _process_transactions_to_holdings,
    _calculate_cash_balances,
    _build_summary_rows,  # More complex, needs mock data/results
    _calculate_aggregate_metrics,  # More complex, needs mock data/results
    safe_sum,
    CASH_SYMBOL_CSV,
    DEFAULT_CURRENCY,
    SHORTABLE_SYMBOLS,
)

# --- Test Fixtures (Sample Data) ---


@pytest.fixture
def sample_dates():
    return [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]


@pytest.fixture
def sample_flows_invest():
    return [-1000.0, -500.0, 1650.0]  # Simple investment, withdrawal, final value


@pytest.fixture
def sample_flows_all_neg():
    return [-1000.0, -500.0, -100.0]


@pytest.fixture
def sample_flows_all_pos():
    return [1000.0, 500.0, 100.0]


@pytest.fixture
def sample_flows_first_pos():
    return [100.0, -50.0, -60.0]


@pytest.fixture
def sample_current_fx_vs_usd():
    # Rates relative to USD (Other Currency per 1 USD)
    return {
        "USD": 1.0,
        "EUR": 0.92,  # 0.92 EUR per 1 USD
        "JPY": 145.0,  # 145 JPY per 1 USD
        "THB": 34.5,  # 34.5 THB per 1 USD
    }


@pytest.fixture
def sample_account_map():
    return {"AccountA": "USD", "AccountB": "EUR", "AccountC": "JPY", "SET": "THB"}


@pytest.fixture
def sample_transactions_csv_string():
    return """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount, Fees,Investment Account,Split Ratio (new shares per old share),Note
"Jan 05, 2023",Buy,AAPL,10,150.00,,5.00,AccountA,,Buy 10 AAPL
"Mar 15, 2023",Dividend,AAPL,,,5.00,0.00,AccountA,,Dividend
"Jun 01, 2023",Sell,AAPL,5,170.00,,5.00,AccountA,,Sell 5 AAPL
"Jul 01, 2023",Buy,MSFT,20,300.00,,5.00,AccountB,,Buy 20 MSFT EUR
"Aug 01, 2023",Deposit,$CASH,1000,,,0.00,AccountA,,Cash Deposit
"Aug 15, 2023",Withdrawal,$CASH,200,,,0.00,AccountA,,Cash Withdrawal
"Sep 01, 2023",Buy,GOOGL,5,100.00,,1.00,AccountC,,Buy 5 GOOGL JPY
"Oct 01, 2023",Split,AAPL,,,,,,2,"2:1 Split"
"Nov 01, 2023",Sell,AAPL,6,180.00,,5.00,AccountA,,Sell 6 AAPL Post Split
"Dec 01, 2023",Fees,$CASH,,,,-10.00,AccountA,,Bank Fee
"Dec 31, 2023",Buy,BAD_DATE,,,10,10,AccountA,,Invalid Date Row - Should be ignored
"Jan 01, 2024",Buy,NVDA,10,BAD_PRICE,,1.00,AccountA,,Invalid Price Row - Should be ignored
"Jan 10, 2024",Buy,MSFT,10,310.00,,5.00,AccountB,,Buy 10 MSFT EUR
"Feb 01, 2024",Buy,XYZ,5,50,,1,UNKNOWN_ACC,, Test Unknown Account
"""


@pytest.fixture
def sample_transactions_df(sample_transactions_csv_string, sample_account_map):
    # Use StringIO to simulate reading from a file
    # Note: Error/Warning flags are ignored here, focus is on the returned DataFrame
    df, _, _, _, _, _ = _load_and_clean_transactions(
        StringIO(sample_transactions_csv_string),
        sample_account_map,
        "USD",  # Default currency for this test
    )
    # Ensure the DataFrame is returned, even if None (pytest will handle assertion)
    return df if df is not None else pd.DataFrame()


# --- Test Cases ---


# --- Test safe_sum ---
def test_safe_sum_basic():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.1, 2.2, np.nan]})
    assert safe_sum(df, "A") == pytest.approx(6.0)
    assert safe_sum(df, "B") == pytest.approx(3.3)


def test_safe_sum_all_nan():
    df = pd.DataFrame({"A": [np.nan, np.nan]})
    assert safe_sum(df, "A") == pytest.approx(0.0)


def test_safe_sum_empty():
    df = pd.DataFrame({"A": []})
    assert safe_sum(df, "A") == pytest.approx(0.0)


def test_safe_sum_missing_col():
    df = pd.DataFrame({"A": [1, 2]})
    assert safe_sum(df, "B") == pytest.approx(0.0)


def test_safe_sum_non_numeric():
    df = pd.DataFrame({"A": [1, "text", 3]})
    # safe_sum attempts conversion, 'text' becomes NaN -> 0
    assert safe_sum(df, "A") == pytest.approx(4.0)


# --- Test calculate_npv ---
def test_calculate_npv(sample_dates, sample_flows_invest):
    rate = 0.10  # 10% discount rate
    # Expected: -1000 / (1.1)^0 + -500 / (1.1)^0.5 + 1650 / (1.1)^1
    # Note: This assumes days/365.0 for years. Let's calc manually for check
    days1 = (sample_dates[1] - sample_dates[0]).days
    days2 = (sample_dates[2] - sample_dates[0]).days
    exp_npv = (
        -1000
        + (-500 / (1 + rate) ** (days1 / 365.0))
        + (1650 / (1 + rate) ** (days2 / 365.0))
    )
    assert calculate_npv(rate, sample_dates, sample_flows_invest) == pytest.approx(
        exp_npv
    )


def test_calculate_npv_zero_rate(sample_dates, sample_flows_invest):
    # With 0 rate, NPV is just the sum of flows
    assert calculate_npv(0.0, sample_dates, sample_flows_invest) == pytest.approx(
        sum(sample_flows_invest)
    )


def test_calculate_npv_high_rate(sample_dates, sample_flows_invest):
    rate = 0.50  # 50% discount rate
    days1 = (sample_dates[1] - sample_dates[0]).days
    days2 = (sample_dates[2] - sample_dates[0]).days
    exp_npv = (
        -1000
        + (-500 / (1 + rate) ** (days1 / 365.0))
        + (1650 / (1 + rate) ** (days2 / 365.0))
    )
    assert calculate_npv(rate, sample_dates, sample_flows_invest) == pytest.approx(
        exp_npv
    )


def test_calculate_npv_empty():
    assert calculate_npv(0.1, [], []) == 0.0


def test_calculate_npv_invalid_rate(sample_dates, sample_flows_invest):
    assert pd.isna(calculate_npv(np.nan, sample_dates, sample_flows_invest))
    assert pd.isna(calculate_npv(np.inf, sample_dates, sample_flows_invest))
    assert pd.isna(calculate_npv(-1.0, sample_dates, sample_flows_invest))
    assert pd.isna(calculate_npv(-2.0, sample_dates, sample_flows_invest))


def test_calculate_npv_mismatched_lengths(sample_dates, sample_flows_invest):
    with pytest.raises(ValueError):
        calculate_npv(0.1, sample_dates[:2], sample_flows_invest)
    with pytest.raises(ValueError):
        calculate_npv(0.1, sample_dates, sample_flows_invest[:2])


# --- Test calculate_irr ---
def test_calculate_irr_standard(sample_dates, sample_flows_invest):
    # Find rate where NPV is approx 0 for the sample flows
    irr = calculate_irr(sample_dates, sample_flows_invest)
    assert isinstance(irr, float)
    assert pd.notna(irr)
    # Check if the calculated IRR yields NPV close to zero
    assert calculate_npv(irr, sample_dates, sample_flows_invest) == pytest.approx(
        0.0, abs=1e-5
    )


def test_calculate_irr_zero_result():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 100]  # 0% return
    assert calculate_irr(dates, flows) == pytest.approx(0.0, abs=1e-6)


def test_calculate_irr_positive_result():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]  # 10% return
    assert calculate_irr(dates, flows) == pytest.approx(0.10, abs=1e-6)


def test_calculate_irr_negative_result():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 90]  # -10% return
    assert calculate_irr(dates, flows) == pytest.approx(-0.10, abs=1e-6)


def test_calculate_irr_invalid_flows(
    sample_dates, sample_flows_all_neg, sample_flows_all_pos, sample_flows_first_pos
):
    assert pd.isna(calculate_irr(sample_dates, sample_flows_all_neg))
    assert pd.isna(calculate_irr(sample_dates, sample_flows_all_pos))
    assert pd.isna(
        calculate_irr(sample_dates, sample_flows_first_pos)
    )  # First flow must be negative


def test_calculate_irr_insufficient_data():
    assert pd.isna(calculate_irr([date(2023, 1, 1)], [-100]))


def test_calculate_irr_unsorted_dates():
    dates = [date(2024, 1, 1), date(2023, 1, 1)]
    flows = [-100, 110]
    assert pd.isna(calculate_irr(dates, flows))  # Should fail validation


# --- Test get_conversion_rate ---
def test_get_conversion_rate_same_currency(sample_current_fx_vs_usd):
    assert get_conversion_rate("USD", "USD", sample_current_fx_vs_usd) == 1.0
    assert get_conversion_rate("EUR", "EUR", sample_current_fx_vs_usd) == 1.0


def test_get_conversion_rate_direct(sample_current_fx_vs_usd):
    # USD -> EUR (Expect: EUR per USD)
    assert get_conversion_rate("USD", "EUR", sample_current_fx_vs_usd) == pytest.approx(
        0.92
    )
    # USD -> JPY (Expect: JPY per USD)
    assert get_conversion_rate("USD", "JPY", sample_current_fx_vs_usd) == pytest.approx(
        145.0
    )


def test_get_conversion_rate_inverse(sample_current_fx_vs_usd):
    # EUR -> USD (Expect: USD per EUR = 1 / (EUR per USD))
    assert get_conversion_rate("EUR", "USD", sample_current_fx_vs_usd) == pytest.approx(
        1 / 0.92
    )
    # JPY -> USD (Expect: USD per JPY = 1 / (JPY per USD))
    assert get_conversion_rate("JPY", "USD", sample_current_fx_vs_usd) == pytest.approx(
        1 / 145.0
    )


def test_get_conversion_rate_cross(sample_current_fx_vs_usd):
    # EUR -> JPY (Expect: (JPY per USD) / (EUR per USD))
    assert get_conversion_rate("EUR", "JPY", sample_current_fx_vs_usd) == pytest.approx(
        145.0 / 0.92
    )
    # THB -> EUR (Expect: (EUR per USD) / (THB per USD))
    assert get_conversion_rate("THB", "EUR", sample_current_fx_vs_usd) == pytest.approx(
        0.92 / 34.5
    )


def test_get_conversion_rate_missing(sample_current_fx_vs_usd):
    # Missing GBP in dict
    assert (
        get_conversion_rate("USD", "GBP", sample_current_fx_vs_usd) == 1.0
    )  # Fallback
    assert (
        get_conversion_rate("GBP", "EUR", sample_current_fx_vs_usd) == 1.0
    )  # Fallback


def test_get_conversion_rate_invalid_input(sample_current_fx_vs_usd):
    assert get_conversion_rate("USD", None, sample_current_fx_vs_usd) == 1.0
    assert get_conversion_rate(None, "EUR", sample_current_fx_vs_usd) == 1.0
    assert get_conversion_rate("USD", "EUR", None) == 1.0


# --- Test _load_and_clean_transactions ---
def test_load_clean_transactions(sample_transactions_csv_string, sample_account_map):
    df, _, ignored_indices, ignored_reasons, has_errors, has_warnings = (
        _load_and_clean_transactions(
            StringIO(sample_transactions_csv_string), sample_account_map, "USD"
        )
    )
    assert (
        df is not None
    ), "DataFrame should not be None after successful load and clean"  # Add check for None df
    assert not has_errors
    assert has_warnings  # Should have warnings for ignored rows
    # --- MODIFIED EXPECTATIONS ---
    # Original indices 0-13 were loaded.
    # Row with original index 10 (BAD_DATE) should be dropped.
    # Row with original index 11 (BAD_PRICE) should be dropped.
    expected_remaining_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13}
    actual_remaining_indices = set(df["original_index"])
    assert actual_remaining_indices == expected_remaining_indices
    # Check the reasons based on the 0-based original index
    assert (
        ignored_reasons.get(10) == "Missing Qty/Price Stock"
    )  # Orig idx 10 (BAD_DATE) likely has NaN Qty/Price too
    assert (
        ignored_reasons.get(11) == "Missing Qty/Price Stock"
    )  # Orig idx 11 (BAD_PRICE) has NaN Price
    # --- END MODIFICATION ---

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert pd.api.types.is_numeric_dtype(df["Quantity"])
    assert pd.api.types.is_numeric_dtype(df["Price/Share"])
    assert pd.api.types.is_numeric_dtype(df["Total Amount"])
    assert pd.api.types.is_numeric_dtype(df["Commission"])
    assert pd.api.types.is_numeric_dtype(df["Split Ratio"])

    # Check specific values and local currency
    assert df.iloc[0]["Symbol"] == "AAPL"
    assert df.iloc[0]["Account"] == "AccountA"
    assert df.iloc[0]["Local Currency"] == "USD"
    assert df.iloc[0]["Quantity"] == 10.0
    assert df.iloc[0]["Commission"] == 5.0

    assert df.iloc[3]["Symbol"] == "MSFT"
    assert df.iloc[3]["Account"] == "AccountB"
    assert df.iloc[3]["Local Currency"] == "EUR"  # From map
    assert df.iloc[3]["Quantity"] == 20.0

    assert df.iloc[4]["Symbol"] == CASH_SYMBOL_CSV
    assert df.iloc[4]["Account"] == "AccountA"
    assert df.iloc[4]["Local Currency"] == "USD"
    assert df.iloc[4]["Quantity"] == 1000.0  # Cash deposit quantity

    # Check unknown account currency assignment
    assert df.iloc[-1]["Symbol"] == "XYZ"
    assert df.iloc[-1]["Account"] == "UNKNOWN_ACC"
    assert df.iloc[-1]["Local Currency"] == "USD"  # Default


# --- Test _calculate_cash_balances ---
def test_calculate_cash_balances(sample_transactions_df):
    # Filter for relevant columns for clarity (though function doesn't require it)
    cash_tx = sample_transactions_df[
        sample_transactions_df["Symbol"] == CASH_SYMBOL_CSV
    ]
    cash_summary, has_errors, has_warnings = _calculate_cash_balances(cash_tx, "USD")

    assert not has_errors
    assert not has_warnings
    assert "AccountA" in cash_summary
    # AccountA: Deposit 1000, Withdrawal 200, Fee -10 -> Net Qty = 800, Comm = -10
    assert cash_summary["AccountA"]["qty"] == pytest.approx(
        1000.0 - 200.0
    )  # Qty reflects balance change
    assert cash_summary["AccountA"]["commissions"] == pytest.approx(
        -10.0
    )  # Fees charged
    assert cash_summary["AccountA"]["dividends"] == pytest.approx(0.0)
    assert cash_summary["AccountA"]["currency"] == "USD"


# --- Test _process_transactions_to_holdings ---
def test_process_transactions_holdings_basic(sample_transactions_df):
    stock_tx = sample_transactions_df[
        sample_transactions_df["Symbol"] != CASH_SYMBOL_CSV
    ]
    # --- >> ADD PRINT << ---
    print("\nDEBUG Holdings Test: Input stock_tx original_indices:")
    print(sorted(list(stock_tx["original_index"])))
    # --- >> END ADD << ---
    holdings, realized, dividends, commissions, ignored_i, ignored_r, has_warnings = (
        _process_transactions_to_holdings(stock_tx, "USD", SHORTABLE_SYMBOLS)
    )

    assert not has_warnings  # Expect no warnings for this valid subset
    assert ("AAPL", "AccountA") in holdings
    assert ("MSFT", "AccountB") in holdings
    assert ("GOOGL", "AccountC") in holdings

    # AAPL Calculations (AccountA, USD)
    # Buy 10 @ 150 (Cost 1505) -> total_buy_cost = 1505, cum_inv = 1505
    # Div 5 (Comm 0) -> dividends = 5, cum_inv = 1505 = 1505
    # Sell 5 @ 170 (Comm 5) -> Proceeds = 5*170-5 = 845. CostSold = 5 * (1505/10) = 752.5. Gain = 845-752.5=92.5. qty=5, total_cost=752.5, cum_inv = 1500 - 845 = 655
    # Split 2:1 -> qty = 10, total_cost=752.5
    # Sell 6 @ 180 (Comm 5) -> Proceeds = 6*180-5 = 1075. CostSold = 6 * (752.5/10) = 451.5. Gain = 1075-451.5=623.5. qty=4, total_cost=752.5-451.5=301.0, cum_inv = 655 - 1075 = -420
    aapl_hold = holdings[("AAPL", "AccountA")]
    assert aapl_hold["qty"] == pytest.approx(4.0)
    assert aapl_hold["total_cost_local"] == pytest.approx(301.0)  # Cost basis remaining
    assert aapl_hold["realized_gain_local"] == pytest.approx(92.5 + 623.5)
    assert aapl_hold["dividends_local"] == pytest.approx(5.0)
    assert aapl_hold["commissions_local"] == pytest.approx(
        5.0 + 5.0 + 5.0
    )  # 3 transactions with fees
    assert aapl_hold["total_buy_cost_local"] == pytest.approx(
        1505.0
    )  # Only the first buy cost
    assert aapl_hold["cumulative_investment_local"] == pytest.approx(
        -415.0
    )  # Final cumulative investment
    assert aapl_hold["local_currency"] == "USD"

    # MSFT Calculations (AccountB, EUR)
    # Buy 20 @ 300 (Cost 6005) -> total_buy_cost = 6005, cum_inv = 6005
    # Buy 10 @ 310 (Cost 3105) -> total_buy_cost = 6005+3105=9110, cum_inv=6005+3105=9110
    msft_hold = holdings[("MSFT", "AccountB")]
    assert msft_hold["qty"] == pytest.approx(30.0)
    assert msft_hold["total_cost_local"] == pytest.approx(6005.0 + 3105.0)
    assert msft_hold["realized_gain_local"] == pytest.approx(0.0)
    assert msft_hold["dividends_local"] == pytest.approx(0.0)
    assert msft_hold["commissions_local"] == pytest.approx(5.0 + 5.0)
    assert msft_hold["total_buy_cost_local"] == pytest.approx(6005.0 + 3105.0)
    assert msft_hold["cumulative_investment_local"] == pytest.approx(6005.0 + 3105.0)
    assert msft_hold["local_currency"] == "EUR"

    # Check overall aggregations (ensure keys exist for relevant currencies)
    assert realized["USD"] == pytest.approx(92.5 + 623.5)
    assert realized.get("EUR", 0.0) == pytest.approx(0.0)
    assert realized.get("JPY", 0.0) == pytest.approx(0.0)

    assert dividends["USD"] == pytest.approx(5.0)
    assert dividends.get("EUR", 0.0) == pytest.approx(0.0)

    assert commissions["USD"] == pytest.approx(5.0 + 5.0 + 5.0 + 1.0)  # AAPL only
    assert commissions["EUR"] == pytest.approx(5.0 + 5.0)  # MSFT only
    assert commissions["JPY"] == pytest.approx(1.0)  # GOOGL only


# --- More complex tests for _build_summary_rows and _calculate_aggregate_metrics ---
# These would require mocking current_stock_data, current_fx_rates_vs_usd
# and potentially using the outputs from the previous test helpers.
# Example structure (requires significant mocking setup):

# @pytest.fixture
# def mock_current_data():
#     stock_data = {
#         'AAPL': {'price': 190.0, 'change': 1.5, 'changesPercentage': 0.79 , 'previousClose': 188.5},
#         'MSFT': {'price': 330.0, 'change': -2.0, 'changesPercentage': -0.60, 'previousClose': 332.0},
#         'GOOGL': {'price': 110.0, 'change': 0.5, 'changesPercentage': 0.45, 'previousClose': 109.5},
#          # Add entries for other symbols if needed by tests
#     }
#     # Assume USD based rates: OTHER CURRENCY per 1 USD
#     fx_rates = { 'USD': 1.0, 'EUR': 0.9, 'JPY': 140.0, 'THB': 35.0 }
#     return stock_data, fx_rates

# def test_build_summary_rows_structure(sample_transactions_df, mock_current_data, sample_account_map):
#     # Run processing steps first
#     stock_tx = sample_transactions_df[sample_transactions_df['Symbol'] != CASH_SYMBOL_CSV]
#     cash_tx = sample_transactions_df[sample_transactions_df['Symbol'] == CASH_SYMBOL_CSV]
#     holdings, _, _, _, _, _, _ = _process_transactions_to_holdings(stock_tx, 'USD', SHORTABLE_SYMBOLS)
#     cash_summary, _, _ = _calculate_cash_balances(cash_tx, 'USD')
#     stock_data, fx_rates = mock_current_data
#     report_date = date(2024, 1, 15) # Example report date
#     display_currency = 'EUR'

#     summary_rows, acc_mv_local, acc_curr_map, has_errors, has_warnings = _build_summary_rows(
#         holdings, cash_summary, stock_data, fx_rates, display_currency, 'USD',
#         sample_transactions_df, report_date, SHORTABLE_SYMBOLS, set() # No excluded symbols in this test
#     )

#     assert not has_errors
#     assert isinstance(summary_rows, list)
#     assert len(summary_rows) > 0 # Expect rows for AAPL, MSFT, GOOGL, Cash

#     # Find specific rows and check basic structure/values
#     aapl_row = next((r for r in summary_rows if r['Symbol'] == 'AAPL' and r['Account'] == 'AccountA'), None)
#     assert aapl_row is not None
#     assert aapl_row['Local Currency'] == 'USD'
#     assert f'Market Value ({display_currency})' in aapl_row
#     assert f'Total Buy Cost ({display_currency})' in aapl_row # Check new column
#     assert pd.notna(aapl_row[f'Market Value ({display_currency})'])

#     msft_row = next((r for r in summary_rows if r['Symbol'] == 'MSFT' and r['Account'] == 'AccountB'), None)
#     assert msft_row is not None
#     assert msft_row['Local Currency'] == 'EUR'
#     assert pd.notna(msft_row[f'Market Value ({display_currency})'])

#     cash_row = next((r for r in summary_rows if r['Symbol'] == CASH_SYMBOL_CSV and r['Account'] == 'AccountA'), None)
#     assert cash_row is not None
#     assert cash_row['Local Currency'] == 'USD'
#     assert pd.notna(cash_row[f'Market Value ({display_currency})'])

# def test_calculate_aggregate_metrics(sample_transactions_df, mock_current_data, sample_account_map):
#      # Similar setup as test_build_summary_rows... run preceding steps
#      # ...
#      # summary_rows, _, _, _, _ = _build_summary_rows(...)
#      # full_summary_df = pd.DataFrame(summary_rows)
#      # Convert types...

#      # _, account_metrics, has_errors, has_warnings = _calculate_aggregate_metrics(
#      #     full_summary_df, 'EUR', sample_transactions_df, date(2024, 1, 15)
#      # )
#      # assert not has_errors
#      # assert 'AccountA' in account_metrics
#      # assert 'AccountB' in account_metrics
#      # assert pd.notna(account_metrics['AccountA']['total_market_value_display'])
#      # assert pd.notna(account_metrics['AccountA']['total_return_pct']) # Should use buy cost now
#      pass # Requires full implementation with mocks


# --- Add more tests for edge cases, different transaction types, etc. ---

# --- END OF FILE test_portfolio_logic.py ---
