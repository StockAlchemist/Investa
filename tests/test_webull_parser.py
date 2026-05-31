"""Unit tests for the Webull monthly-statement parser in server.pdf_parser.

These exercise the pure row/table classification helpers with synthetic table
rows mirroring the layouts in a real Webull (Thailand) Monthly Account Statement,
so they don't require a sample PDF on disk.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from server.pdf_parser import (  # noqa: E402
    _webull_date,
    _webull_float,
    _webull_classify_table,
    _webull_cashflow_row,
    _webull_trade_row,
    _webull_dividend_rows,
    _webull_tag_reinvestments,
)


def test_webull_date_is_day_month_year():
    assert _webull_date("01/04/2026") == "2026-04-01"
    assert _webull_date("09/04/2026") == "2026-04-09"
    assert _webull_date("not a date") is None
    assert _webull_date("") is None


def test_webull_float_handles_negatives_and_separators():
    assert _webull_float("12,722.88") == 12722.88
    assert _webull_float("(332.12)") == -332.12
    assert _webull_float("5.00-") == -5.0
    assert _webull_float("0.01") == 0.01


def test_classify_skips_snapshot_and_accrual_tables():
    net_value = [["", "Total Cash", "Market Value (Stocks)", "Total Value"]]
    cash_report = [["", "HKD", "CNH", "THB", "USD", "Total(THB)"]]
    holdings = [["Symbol & Name", "Quantity", "Average Price", "Market Value", "Unrealized P/L"]]
    accrued = [["Symbol & Name", "Ex-Date", "Pay Date", "Quantity", "Gross Amount", "Net Amount"]]
    for table in (net_value, cash_report, holdings, accrued):
        assert _webull_classify_table(table) == "skip"


def test_classify_interest_trades_and_dividend_tables():
    interest = [["Date", "Description", "Currency", "Amount"]]
    trades = [["Date", "Symbol & Name", "Side", "Quantity", "Price", "Amount", "Currency"]]
    dividends = [["Posting Date", "Description", "Currency", "Gross Amount",
                  "Withholding Tax", "Net Amount", "Status"]]
    assert _webull_classify_table(interest) == "cashflow"
    assert _webull_classify_table(trades) == "trades"
    assert _webull_classify_table(dividends) == "dividends"


def test_classify_currency_exchange_is_skipped():
    fx = [["Date", "Time", "Initial Currency", "Initial Amount",
           "Converted Currency", "Converted Amount", "Converted Rate"]]
    assert _webull_classify_table(fx) == "skip"


def test_cashflow_row_parses_interest_credit():
    row = ["01/04/2026", "Credit - Credit Interest 03/01/2026 to 03/31/2026", "USD", "0.01"]
    tx = _webull_cashflow_row(row, "CTH7812155", user_id=7, default_currency="THB")
    assert tx is not None
    assert tx["Date"] == "2026-04-01"
    assert tx["Type"] == "Interest"
    assert tx["Symbol"] == "$CASH"
    assert tx["Total Amount"] == 0.01
    assert tx["Quantity"] == 0.01
    assert tx["Local Currency"] == "USD"
    assert tx["Account"] == "CTH7812155"
    assert tx["user_id"] == 7


def test_cashflow_row_signs_outflows_negative():
    fee = ["15/04/2026", "Commission Fee", "THB", "20.00"]
    tx = _webull_cashflow_row(fee, "Acct", user_id=1, default_currency="THB")
    assert tx["Type"] == "Fees"
    assert tx["Total Amount"] == -20.0

    withdrawal = ["20/04/2026", "Cash Withdrawal", "THB", "1,000.00"]
    tx = _webull_cashflow_row(withdrawal, "Acct", user_id=1, default_currency="THB")
    assert tx["Type"] == "Withdrawal"
    assert tx["Total Amount"] == -1000.0


def test_cashflow_row_classifies_dividend_and_deposit():
    div = ["08/05/2026", "Cash Dividend MASTERCARD", "USD", "6.52"]
    tx = _webull_cashflow_row(div, "Acct", user_id=1, default_currency="THB")
    assert tx["Type"] == "Dividend"
    assert tx["Total Amount"] == 6.52

    dep = ["02/04/2026", "Deposit - Bank Transfer", "THB", "50,000.00"]
    tx = _webull_cashflow_row(dep, "Acct", user_id=1, default_currency="THB")
    assert tx["Type"] == "Deposit"
    assert tx["Total Amount"] == 50000.0


def test_cashflow_row_skips_total_and_undated_rows():
    assert _webull_cashflow_row(["", "Total", "USD", "0.01"], "A", 1, "THB") is None
    assert _webull_cashflow_row(["Closing Cash", "", "", "80.23"], "A", 1, "THB") is None
    assert _webull_cashflow_row(["no date", "Some text", "", ""], "A", 1, "THB") is None


# Real Webull TRADES table column layout (no currency column).
_TRADE_HEADER = [
    "Symbol & Name", "Trade Date", "Time", "Settlement Date", "Buy/Sell",
    "Quantity", "Traded Price", "Gross Amount", "Net Amount", "Comm/Fee/Tax",
    "VAT", "Exchange", "Remarks", "Status",
]


def test_trade_row_parses_buy_with_correct_columns():
    # From 2025-11.PDF: 9 AMZN @ 220.16 on NASDAQ, zero fees.
    row = ["AMZN AMAZON COM INC", "20/11/2025", "00:12:36,GMT+07", "21/11/2025",
           "BUY", "9", "220.16", "1,981.44", "1,981.44", "0.00", "0.00",
           "NASDAQ", "", ""]
    tx = _webull_trade_row(row, _TRADE_HEADER, "Acct", user_id=3, default_currency="THB")
    assert tx is not None
    assert tx["Type"] == "Buy"
    assert tx["Symbol"] == "AMZN"
    assert tx["Quantity"] == 9.0
    assert tx["Price/Share"] == 220.16
    assert tx["Total Amount"] == 1981.44  # Gross Amount, not Net/commission
    assert tx["Commission"] == 0.0
    # No currency column on Webull trades -> inferred from the NASDAQ exchange.
    assert tx["Local Currency"] == "USD"


def test_trade_row_sums_commission_and_vat():
    row = ["MA MASTERCARD INCORPORATED", "10/02/2026", "22:05:00,GMT+07",
           "11/02/2026", "BUY", "2", "545.13", "1,090.26", "1,092.26", "1.50",
           "0.50", "NYSE", "", ""]
    tx = _webull_trade_row(row, _TRADE_HEADER, "Acct", user_id=3, default_currency="THB")
    assert tx["Commission"] == 2.0  # Comm/Fee/Tax 1.50 + VAT 0.50
    assert tx["Total Amount"] == 1090.26


def test_trade_row_infers_sell_from_negative_quantity():
    row = ["MA MASTERCARD INCORPORATED", "06/04/2026", "21:00:00,GMT+07",
           "08/04/2026", "", "-5", "502.92", "2,514.60", "2,514.60", "0.00",
           "0.00", "NYSE", "", ""]
    tx = _webull_trade_row(row, _TRADE_HEADER, "Acct", user_id=3, default_currency="THB")
    assert tx["Type"] == "Sell"
    assert tx["Symbol"] == "MA"
    assert tx["Quantity"] == 5.0


# Real Webull DIVIDENDS table column layout.
_DIV_HEADER = ["Posting Date", "Description", "Currency", "Gross Amount",
               "Withholding Tax", "Net Amount", "Status"]
_NAME_MAP = {"meta platforms inc": "META", "mastercard incorporated": "MA"}


def test_dividend_rows_split_gross_and_withholding_tax():
    row = ["26/03/2026",
           "META PLATFORMS INC - Cash Div on 5.003 shares - Rec 03/16 /2026 Pay 03/26/2026",
           "USD", "2.63", "-0.39", "2.24", ""]
    out = _webull_dividend_rows(row, _DIV_HEADER, "Webull", user_id=1,
                                default_currency="THB", name_to_symbol=_NAME_MAP)
    assert len(out) == 2
    div, tax = out
    # Gross dividend leg, tied to the ticker via the holdings name map.
    assert div["Type"] == "Dividend"
    assert div["Symbol"] == "META"
    assert div["Total Amount"] == 2.63
    assert div["Quantity"] == 1.0
    assert div["Note"] == "Gross Dividend"
    # Separate withholding-tax leg (positive magnitude).
    assert tax["Type"] == "Tax"
    assert tax["Symbol"] == "META"
    assert tax["Total Amount"] == 0.39
    assert tax["Note"] == "Dividend Tax"


def test_dividend_with_no_withholding_emits_single_leg():
    row = ["09/02/2026", "MASTERCARD INCORPORATED - Cash Div", "USD",
           "7.66", "0.00", "7.66", ""]
    out = _webull_dividend_rows(row, _DIV_HEADER, "Webull", user_id=1,
                                default_currency="THB", name_to_symbol=_NAME_MAP)
    assert len(out) == 1
    assert out[0]["Type"] == "Dividend"
    assert out[0]["Symbol"] == "MA"


def test_reinvestment_buy_is_tagged_and_nets_to_zero():
    # gross 2.63 - WHT 0.39 - reinvest 2.24 = 0
    txns = [
        {"Date": "2026-03-26", "Type": "Dividend", "Symbol": "META",
         "Total Amount": 2.63, "Note": "Gross Dividend"},
        {"Date": "2026-03-26", "Type": "Tax", "Symbol": "META",
         "Total Amount": 0.39, "Note": "Dividend Tax"},
        {"Date": "2026-03-27", "Type": "Buy", "Symbol": "META",
         "Total Amount": 2.24, "Note": "Webull Trade"},
    ]
    out = _webull_tag_reinvestments(txns)
    buy = next(t for t in out if t["Type"] == "Buy")
    assert buy["Note"] == "Dividend Reinvestment"
    net_cash = 2.63 - 0.39 - 2.24
    assert abs(net_cash) < 1e-9


def test_ordinary_buy_not_tagged_as_reinvestment():
    # A same-symbol buy far from the dividend and for an unrelated amount stays a
    # plain trade.
    txns = [
        {"Date": "2026-03-26", "Type": "Dividend", "Symbol": "AMZN",
         "Total Amount": 2.63, "Note": "Gross Dividend"},
        {"Date": "2026-03-04", "Type": "Buy", "Symbol": "AMZN",
         "Total Amount": 4811.52, "Note": "Webull Trade"},
    ]
    out = _webull_tag_reinvestments(txns)
    assert out[1]["Note"] == "Webull Trade"
