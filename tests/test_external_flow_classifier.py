"""Tests for the bookkeeping-mode-independent external-flow classifier.

The classifier is the single source of truth for what counts as an external
portfolio flow in both the TWR engine
(`_calculate_daily_net_cash_flow_vectorized` in portfolio_logic.py) and the
IRR/MWR engine (`get_cash_flows_for_mwr` in finutils.py under
`flow_basis="portfolio"`). It must remain stable so that dheematan (auto-cash
mode) and kitmatan (manual-cash mode with paired settlement rows) see the
same external flows for the same underlying portfolio.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from finutils import is_external_flow_row, compute_external_flow_mask


# --- Per-row classifier --------------------------------------------------


def test_real_ach_deposit_is_external():
    assert is_external_flow_row("$CASH", "Deposit", "ACH from Bank of America") is True
    assert is_external_flow_row("$CASH", "deposit", "Wire received") is True


def test_real_external_withdrawal_is_external():
    assert is_external_flow_row("$CASH", "Withdrawal", "Withdrawal to checking") is True
    assert is_external_flow_row("$CASH", "withdrawal", "") is True
    assert is_external_flow_row("$CASH", "withdrawal", None) is True


def test_auto_generated_deposit_is_internal():
    # dheematan import-tool synthetic per-trade entries
    assert is_external_flow_row("$CASH", "Deposit", "Auto-generated: Cash deposit for SPY buy") is False
    assert is_external_flow_row("$CASH", "Deposit", "auto-generated: cash deposit for AAPL buy") is False


def test_auto_generated_withdrawal_is_internal():
    assert is_external_flow_row("$CASH", "Withdrawal", "Auto-generated: Cash withdrawal from MSFT sell") is False


def test_cash_buy_sell_settlement_is_internal():
    # kitmatan manual-mode settlement rows
    assert is_external_flow_row("$CASH", "Buy", "Auto-generated: Cash received from QQQ sell") is False
    assert is_external_flow_row("$CASH", "Sell", "Auto-generated: Cash settlement for QQQ buy") is False
    # Even without the Auto-gen prefix, $CASH buy/sell are internal settlement
    assert is_external_flow_row("$CASH", "buy", "") is False
    assert is_external_flow_row("$CASH", "sell", None) is False


def test_cash_dividend_interest_is_internal():
    assert is_external_flow_row("$CASH", "Dividend", "Money market dividend") is False
    assert is_external_flow_row("$CASH", "Interest", "T-Bill interest") is False


def test_stock_trades_are_internal():
    assert is_external_flow_row("AAPL", "Buy", "") is False
    assert is_external_flow_row("MSFT", "Sell", "") is False
    assert is_external_flow_row("QQQ", "Dividend", "") is False
    assert is_external_flow_row("VTI", "Tax", "W-8 withholding") is False
    assert is_external_flow_row("AAPL", "Fees", "Commission") is False
    assert is_external_flow_row("SPY", "Transfer", "Account migration") is False


def test_commission_disguised_as_deposit_withdrawal_is_internal():
    # kitmatan-style miscategorization: broker commissions stored as
    # Deposit/Withdrawal with a "Commission for X buy" Note.
    assert is_external_flow_row("$CASH", "Withdrawal", "Commission for MSFT Buy") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Commission Buy AMZN") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fee on buying MA") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fee for trade") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fees on AMZN buy") is False
    # Case-insensitive
    assert is_external_flow_row("$CASH", "Withdrawal", "commission for nflx sell") is False
    # But a Note that just mentions commission elsewhere should still be external
    assert is_external_flow_row("$CASH", "Deposit", "ACH wire net of commissions") is True


def test_non_string_inputs_are_safe():
    # Defensive: don't crash on NaN, ints, None
    assert is_external_flow_row(None, "Deposit", "") is False
    assert is_external_flow_row("$CASH", None, "") is False
    assert is_external_flow_row("$CASH", 123, "") is False
    assert is_external_flow_row("$CASH", "Deposit", 456) is True


# --- Vectorized mask -----------------------------------------------------


def test_vectorized_mask_matches_per_row():
    """The vectorized mask must agree with is_external_flow_row row-by-row."""
    df = pd.DataFrame([
        # External: real ACH
        {"Symbol": "$CASH", "Type": "Deposit", "Note": "ACH from bank"},
        # External: real withdrawal, no note
        {"Symbol": "$CASH", "Type": "Withdrawal", "Note": ""},
        # Internal: auto-generated deposit
        {"Symbol": "$CASH", "Type": "Deposit", "Note": "Auto-generated: Cash deposit for SPY buy"},
        # Internal: $CASH buy/sell settlement
        {"Symbol": "$CASH", "Type": "Buy", "Note": "settlement"},
        {"Symbol": "$CASH", "Type": "Sell", "Note": "settlement"},
        # Internal: $CASH dividend/interest
        {"Symbol": "$CASH", "Type": "Dividend", "Note": "MM div"},
        # Internal: stock trades
        {"Symbol": "AAPL", "Type": "Buy", "Note": ""},
        {"Symbol": "QQQ", "Type": "Sell", "Note": ""},
        {"Symbol": "VTI", "Type": "Dividend", "Note": ""},
        # Edge case: Note is NaN
        {"Symbol": "$CASH", "Type": "Deposit", "Note": float("nan")},
    ])

    expected = [True, True, False, False, False, False, False, False, False, True]
    mask = compute_external_flow_mask(df)
    assert list(mask) == expected, f"got {list(mask)}, expected {expected}"


def test_vectorized_mask_handles_missing_note_column():
    """Missing Note column treated as empty string (no Auto-gen matches)."""
    df = pd.DataFrame([
        {"Symbol": "$CASH", "Type": "Deposit"},
        {"Symbol": "$CASH", "Type": "Buy"},
        {"Symbol": "AAPL", "Type": "Buy"},
    ])
    mask = compute_external_flow_mask(df)
    assert list(mask) == [True, False, False]


def test_vectorized_mask_handles_empty_dataframe():
    df = pd.DataFrame(columns=["Symbol", "Type", "Note"])
    mask = compute_external_flow_mask(df)
    assert len(mask) == 0
    assert mask.dtype == bool
