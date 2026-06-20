"""Unit tests for the IBKR Trades parsing helpers in server.pdf_parser.

IBKR ships the Trades section in two PDF layouts with different column orders
(Activity Statement vs Trade Confirmation Report). These exercise the pure
header/row helpers with synthetic rows mirroring real PDFs, so they don't
require a sample PDF on disk.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from server.pdf_parser import (  # noqa: E402
    _ibkr_trades_colmap,
    _ibkr_trade_from_row,
)

# --- Trade Confirmation Report layout (the one that triggered the bug) ---
TC_HEADER = [
    "Acct ID", "Symbol", "Trade Date/Time", "Settle Date", "Exchange", "Type",
    "Quantity", "Price", "Proceeds", "Comm", "Fee", "Order Type", "Code",
]
TC_SUMMARY = [
    "U13340051", "ASML", "2026-06-02, 09:42:49", "2026-06-03", "-", "SELL",
    "-6", "1,677.0000", "10,062.00", "-1.21", "0.00", "LMT", "C;P",
]
TC_FILL = [
    "U13340051", "ASML", "2026-06-02, 09:42:49", "2026-06-03", "NASDAQ", "SELL",
    "-1", "1,677.0000", "1,677.00", "-1.03", "0.00", "LMT", "C;P",
]
TC_SUBTOTAL = [
    "Total ASML (Sold)", "", "", "", "", "", "-6", "1,677.0000", "10,062.00",
    "-1.21", "0.00", "", "",
]

# --- Activity Statement layout ---
AS_HEADER = [
    "Symbol", "Date/Time", "", "Quantity", "T. Price", "C. Price", "Proceeds",
    "Comm/Fee", "Basis", "Realized P/L", "", "MTM P/L", "Code",
]
AS_ROW = [
    "AAPL", "2026-03-09, 13:07:27", "", "-78", "257.3379", "259.8800",
    "20,072.36", "-1.02", "-14,144.91", "5,926.43", "", "-198.28", "C;P",
]


def test_colmap_trade_confirmation():
    cols = _ibkr_trades_colmap(TC_HEADER)
    assert cols["acct"] == 0
    assert cols["sym"] == 1
    assert cols["date"] == 2  # 'Trade Date/Time', not 'Settle Date' (3)
    assert cols["exchange"] == 4
    assert cols["qty"] == 6
    assert cols["price"] == 7
    assert cols["proceeds"] == 8  # the bug read this as 9 (Comm)
    assert cols["comm"] == 9
    assert cols["fee"] == 10


def test_colmap_activity_statement():
    cols = _ibkr_trades_colmap(AS_HEADER)
    assert cols["sym"] == 0
    assert cols["date"] == 1
    assert cols["qty"] == 3
    assert cols["price"] == 4  # 'T. Price', not 'C. Price' (5)
    assert cols["proceeds"] == 6
    assert cols["comm"] == 7
    assert "exchange" not in cols
    assert "acct" not in cols


def test_trade_confirmation_summary_row():
    """Total Amount is the net proceeds (Proceeds - commission for a Sell), not
    the bare commission. Net matches the manual-add / IBKR-connector convention
    the valuation kernel uses for cash flow."""
    cols = _ibkr_trades_colmap(TC_HEADER)
    txn = _ibkr_trade_from_row(TC_SUMMARY, cols, "IBKR", user_id=1)
    assert txn is not None
    assert txn["Type"] == "Sell"
    assert txn["Symbol"] == "ASML"
    assert txn["Quantity"] == 6.0
    assert txn["Price/Share"] == 1677.0
    assert txn["Total Amount"] == 10060.79  # proceeds 10062.00 net of 1.21 commission
    assert txn["Commission"] == 1.21
    assert txn["Account"] == "U13340051"  # taken from the Acct ID column


def test_trade_confirmation_fill_rows_are_skipped():
    """Per-venue fills duplicate the order summary; they must not be imported."""
    cols = _ibkr_trades_colmap(TC_HEADER)
    assert _ibkr_trade_from_row(TC_FILL, cols, "IBKR", user_id=1) is None


def test_trade_confirmation_subtotal_row_is_skipped():
    cols = _ibkr_trades_colmap(TC_HEADER)
    assert _ibkr_trade_from_row(TC_SUBTOTAL, cols, "IBKR", user_id=1) is None


def test_commission_and_fee_are_combined():
    cols = _ibkr_trades_colmap(TC_HEADER)
    row = list(TC_SUMMARY)
    row[9], row[10] = "-1.00", "-0.50"  # Comm + Fee
    txn = _ibkr_trade_from_row(row, cols, "IBKR", user_id=1)
    assert txn["Commission"] == 1.50


def test_activity_statement_row_imported():
    cols = _ibkr_trades_colmap(AS_HEADER)
    txn = _ibkr_trade_from_row(AS_ROW, cols, "IBKR", user_id=1)
    assert txn is not None
    assert txn["Symbol"] == "AAPL"
    assert txn["Quantity"] == 78.0
    assert txn["Total Amount"] == 20071.34  # proceeds 20072.36 net of 1.02 commission (Sell)
    assert txn["Commission"] == 1.02
    assert txn["Account"] == "IBKR"  # no Acct ID column -> falls back to override
