"""Tests for the bookkeeping-mode-independent external-flow classifier.

The classifier is the single source of truth for what counts as an external
portfolio flow in both the TWR engine
(`_calculate_daily_net_cash_flow_vectorized` in portfolio_logic.py) and the
IRR/MWR engine (`get_cash_flows_for_mwr` in finutils.py under
`flow_basis="portfolio"`).

Convention: "always external" (trading-account view).
  External iff Symbol == $CASH AND Type IN {Deposit, Withdrawal}.
  Note content is NOT inspected. If a row's Note describes an internal cost
  (e.g. broker commission), it must be retyped at the data layer rather than
  filtered here.
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


def test_auto_generated_deposit_is_still_external():
    # Trading-account convention: per-trade synthetic deposits are still flows.
    # The user can think of them as "money wired in to fund this buy".
    assert is_external_flow_row("$CASH", "Deposit", "Auto-generated: Cash deposit for SPY buy") is True
    assert is_external_flow_row("$CASH", "deposit", "auto-generated: cash deposit for AAPL buy") is True


def test_auto_generated_withdrawal_is_still_external():
    assert is_external_flow_row("$CASH", "Withdrawal", "Auto-generated: Cash withdrawal from MSFT sell") is True


def test_cash_buy_sell_settlement_is_internal():
    # $CASH buy/sell (kitmatan-style manual-mode settlement) → not Deposit/Withdrawal → internal.
    assert is_external_flow_row("$CASH", "Buy", "Auto-generated: Cash received from QQQ sell") is False
    assert is_external_flow_row("$CASH", "Sell", "Auto-generated: Cash settlement for QQQ buy") is False
    assert is_external_flow_row("$CASH", "buy", "") is False
    assert is_external_flow_row("$CASH", "sell", None) is False


def test_cash_dividend_interest_is_internal():
    assert is_external_flow_row("$CASH", "Dividend", "Money market dividend") is False
    assert is_external_flow_row("$CASH", "Interest", "T-Bill interest") is False


def test_cash_fees_tax_is_internal():
    # $CASH Fees/Tax (e.g. wire fees, foreign-dividend withholding paid in cash)
    # are internal costs, not external flows.
    assert is_external_flow_row("$CASH", "Fees", "Outgoing wire fee") is False
    assert is_external_flow_row("$CASH", "Tax", "Foreign withholding") is False


def test_stock_trades_are_internal():
    assert is_external_flow_row("AAPL", "Buy", "") is False
    assert is_external_flow_row("MSFT", "Sell", "") is False
    assert is_external_flow_row("QQQ", "Dividend", "") is False
    assert is_external_flow_row("VTI", "Tax", "W-8 withholding") is False
    assert is_external_flow_row("AAPL", "Fees", "Commission") is False
    assert is_external_flow_row("SPY", "Transfer", "Account migration") is False


def test_commission_notes_are_still_external_if_typed_as_deposit_withdrawal():
    # Note prefix is NOT inspected. If your data encodes commissions as
    # Type=Withdrawal, the classifier treats them as external — fix at the
    # data layer by retyping to Type=Fees instead.
    assert is_external_flow_row("$CASH", "Withdrawal", "Commission for MSFT Buy") is True
    assert is_external_flow_row("$CASH", "Withdrawal", "Fee on buying AMZN") is True


def test_note_is_optional():
    # `note` argument is accepted (for back-compat) but ignored.
    assert is_external_flow_row("$CASH", "Deposit") is True
    assert is_external_flow_row("$CASH", "Withdrawal") is True
    assert is_external_flow_row("AAPL", "Buy") is False


def test_non_string_inputs_are_safe():
    # Defensive: don't crash on NaN, ints, None.
    assert is_external_flow_row(None, "Deposit", "") is False
    assert is_external_flow_row("$CASH", None, "") is False
    assert is_external_flow_row("$CASH", 123, "") is False


# --- Vectorized mask -----------------------------------------------------


def test_vectorized_mask_matches_per_row():
    """The vectorized mask must agree with is_external_flow_row row-by-row."""
    df = pd.DataFrame([
        # External: real ACH
        {"Symbol": "$CASH", "Type": "Deposit", "Note": "ACH from bank"},
        # External: real withdrawal, no note
        {"Symbol": "$CASH", "Type": "Withdrawal", "Note": ""},
        # External: auto-generated deposit (still external under always-external convention)
        {"Symbol": "$CASH", "Type": "Deposit", "Note": "Auto-generated: Cash deposit for SPY buy"},
        # Internal: $CASH buy/sell settlement (wrong Type)
        {"Symbol": "$CASH", "Type": "Buy", "Note": "settlement"},
        {"Symbol": "$CASH", "Type": "Sell", "Note": "settlement"},
        # Internal: $CASH dividend/interest/fees/tax (wrong Type)
        {"Symbol": "$CASH", "Type": "Dividend", "Note": "MM div"},
        {"Symbol": "$CASH", "Type": "Fees", "Note": "Wire fee"},
        # Internal: stock trades
        {"Symbol": "AAPL", "Type": "Buy", "Note": ""},
        {"Symbol": "QQQ", "Type": "Sell", "Note": ""},
        {"Symbol": "VTI", "Type": "Dividend", "Note": ""},
        # External: Withdrawal with commission-like note — still external (note ignored)
        {"Symbol": "$CASH", "Type": "Withdrawal", "Note": "Commission for MSFT Buy"},
    ])

    expected = [True, True, True, False, False, False, False, False, False, False, True]
    mask = compute_external_flow_mask(df)
    assert list(mask) == expected, f"got {list(mask)}, expected {expected}"


def test_vectorized_mask_handles_missing_note_column():
    """Missing Note column is fine; classifier doesn't depend on it."""
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
