"""Tests for the bookkeeping-mode-independent external-flow classifier.

The classifier is the single source of truth for what counts as an external
portfolio flow in both the TWR engine
(`_calculate_daily_net_cash_flow_vectorized` in portfolio_logic.py) and the
IRR/MWR engine (`get_cash_flows_for_mwr` in finutils.py under
`flow_basis="portfolio"`).

Convention: GIPS / professional standard.
  An external flow is a REAL ACH/wire/check between the investor's outside
  funds and the brokerage account. Stock trades, dividends, interest, taxes,
  fees, and per-trade $CASH settlement rows are INTERNAL.

  External iff
    Symbol == $CASH
    AND Type IN {Deposit, Withdrawal}
    AND Note does NOT start with "Auto-generated:"  (import-tool synthetic
        per-trade entries)
    AND Note does NOT start with broker-cost prefixes ("Commission",
        "Fee on", "Fee for", "Fees on", "Fees for", "Comm ", "Comm.")
        (commissions/fees miscategorized as Deposit/Withdrawal).
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
    # Import-tool synthetic per-trade entry — internal under GIPS convention.
    assert is_external_flow_row("$CASH", "Deposit", "Auto-generated: Cash deposit for SPY buy") is False
    assert is_external_flow_row("$CASH", "Deposit", "auto-generated: cash deposit for AAPL buy") is False


def test_auto_generated_withdrawal_is_internal():
    assert is_external_flow_row("$CASH", "Withdrawal", "Auto-generated: Cash withdrawal from MSFT sell") is False


def test_cash_buy_sell_settlement_is_internal():
    # $CASH buy/sell (manual-mode settlement) — Type not in {Deposit, Withdrawal}.
    assert is_external_flow_row("$CASH", "Buy", "Auto-generated: Cash received from QQQ sell") is False
    assert is_external_flow_row("$CASH", "Sell", "Auto-generated: Cash settlement for QQQ buy") is False
    assert is_external_flow_row("$CASH", "buy", "") is False
    assert is_external_flow_row("$CASH", "sell", None) is False


def test_cash_dividend_interest_is_internal():
    assert is_external_flow_row("$CASH", "Dividend", "Money market dividend") is False
    assert is_external_flow_row("$CASH", "Interest", "T-Bill interest") is False


def test_cash_fees_tax_is_internal():
    # $CASH Fees/Tax (wire fees, foreign withholding etc.) are internal costs.
    assert is_external_flow_row("$CASH", "Fees", "Outgoing wire fee") is False
    assert is_external_flow_row("$CASH", "Tax", "Foreign withholding") is False


def test_stock_trades_are_internal():
    assert is_external_flow_row("AAPL", "Buy", "") is False
    assert is_external_flow_row("MSFT", "Sell", "") is False
    assert is_external_flow_row("QQQ", "Dividend", "") is False
    assert is_external_flow_row("VTI", "Tax", "W-8 withholding") is False
    assert is_external_flow_row("AAPL", "Fees", "Commission") is False
    assert is_external_flow_row("SPY", "Transfer", "Account migration") is False


def test_commission_disguised_as_withdrawal_is_internal():
    # Broker commissions miscategorized as $CASH Withdrawal by some importers.
    assert is_external_flow_row("$CASH", "Withdrawal", "Commission for MSFT Buy") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Commission Buy AMZN") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fee on buying MA") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fee for trade") is False
    assert is_external_flow_row("$CASH", "Withdrawal", "Fees on AMZN buy") is False
    # Case-insensitive
    assert is_external_flow_row("$CASH", "Withdrawal", "commission for nflx sell") is False
    # But a Note that just mentions commission elsewhere is still external
    assert is_external_flow_row("$CASH", "Deposit", "ACH wire net of commissions") is True


def test_non_string_inputs_are_safe():
    assert is_external_flow_row(None, "Deposit", "") is False
    assert is_external_flow_row("$CASH", None, "") is False
    assert is_external_flow_row("$CASH", 123, "") is False
    # Non-string note is coerced to str, then prefix-checked
    assert is_external_flow_row("$CASH", "Deposit", 456) is True


# --- Vectorized mask -----------------------------------------------------


def test_vectorized_mask_matches_per_row():
    df = pd.DataFrame([
        # External: real ACH
        {"Symbol": "$CASH", "Type": "Deposit", "Note": "ACH from bank"},
        # External: real withdrawal, no note
        {"Symbol": "$CASH", "Type": "Withdrawal", "Note": ""},
        # Internal: auto-generated synthetic per-trade
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
        # Internal: commission disguised as Withdrawal
        {"Symbol": "$CASH", "Type": "Withdrawal", "Note": "Commission for MSFT Buy"},
        # Edge case: NaN note
        {"Symbol": "$CASH", "Type": "Deposit", "Note": float("nan")},
    ])

    expected = [True, True, False, False, False, False, False, False, False, False, False, True]
    mask = compute_external_flow_mask(df)
    assert list(mask) == expected, f"got {list(mask)}, expected {expected}"


def test_vectorized_mask_handles_missing_note_column():
    """Missing Note column is fine; no rows match Auto-gen/Commission prefixes."""
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
