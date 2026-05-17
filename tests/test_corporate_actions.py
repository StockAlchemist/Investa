"""
Unit tests pinning the arithmetic for corporate-action helpers.

These are pure-function tests, no portfolio engine required. They lock in the
semantics of return-of-capital and stock-dividend computations before those
helpers get wired into the JIT'd dispatchers in portfolio_logic.py.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from corporate_actions import (  # noqa: E402
    apply_return_of_capital,
    apply_stock_dividend,
    deduplicate_split_transactions,
    RESERVED_CORPORATE_ACTION_TYPES,
    SUPPORTED_TYPES,
)
import pandas as pd  # noqa: E402


# ---------- return of capital ----------

def test_return_of_capital_partial_reduces_basis():
    qty, cost, gain = apply_return_of_capital(
        current_qty=100.0, current_cost=10000.0, cash_distributed=500.0
    )
    assert qty == 100.0
    assert cost == 9500.0
    assert gain == 0.0


def test_return_of_capital_full_zeroes_basis():
    qty, cost, gain = apply_return_of_capital(
        current_qty=100.0, current_cost=10000.0, cash_distributed=10000.0
    )
    assert qty == 100.0
    assert cost == 0.0
    assert gain == 0.0


def test_return_of_capital_excess_becomes_gain():
    # Distribution above remaining basis — the overshoot is a realised gain.
    qty, cost, gain = apply_return_of_capital(
        current_qty=100.0, current_cost=400.0, cash_distributed=1000.0
    )
    assert qty == 100.0
    assert cost == 0.0
    assert gain == 600.0


def test_return_of_capital_with_no_position():
    # Edge case: an ROC row arrives for a position that's already been sold.
    qty, cost, gain = apply_return_of_capital(
        current_qty=0.0, current_cost=0.0, cash_distributed=100.0
    )
    assert qty == 0.0
    assert cost == 0.0
    assert gain == 100.0


# ---------- stock dividend ----------

def test_stock_dividend_increases_qty_keeps_cost():
    qty, cost = apply_stock_dividend(
        current_qty=100.0, current_cost=10000.0, shares_received=5.0
    )
    assert qty == 105.0
    assert cost == 10000.0  # total basis unchanged


def test_stock_dividend_implicit_per_share_cost():
    # 100 shares @ $100/sh → $10,000 basis. 10-share stock dividend → 110 shares @ ~$90.91.
    qty, cost = apply_stock_dividend(
        current_qty=100.0, current_cost=10000.0, shares_received=10.0
    )
    assert qty == 110.0
    assert cost == 10000.0
    assert cost / qty == 10000.0 / 110.0  # ≈ $90.91 per share


def test_stock_dividend_zero_shares_noop():
    qty, cost = apply_stock_dividend(100.0, 10000.0, 0.0)
    assert (qty, cost) == (100.0, 10000.0)


def test_stock_dividend_on_empty_position_noop():
    qty, cost = apply_stock_dividend(0.0, 0.0, 5.0)
    assert (qty, cost) == (0.0, 0.0)


# ---------- type registry guardrails ----------

def test_supported_and_reserved_types_dont_overlap():
    # A type can be supported OR reserved-for-future-work, not both.
    assert SUPPORTED_TYPES.isdisjoint(RESERVED_CORPORATE_ACTION_TYPES)


def test_known_corporate_action_types_present():
    assert "return of capital" in RESERVED_CORPORATE_ACTION_TYPES
    assert "stock dividend" in RESERVED_CORPORATE_ACTION_TYPES
    assert "spin off" in RESERVED_CORPORATE_ACTION_TYPES


# ---------- split dedup (regression guard for the function moved in #4) ----------

def test_deduplicate_split_keeps_all_accounts_row():
    df = pd.DataFrame([
        {"original_index": 0, "Type": "buy",   "Symbol": "AAPL", "Account": "Brokerage", "Date": "2025-01-01", "Split Ratio": 1.0},
        {"original_index": 1, "Type": "split", "Symbol": "AAPL", "Account": "All Accounts", "Date": "2025-06-01", "Split Ratio": 2.0},
        {"original_index": 2, "Type": "split", "Symbol": "AAPL", "Account": "Brokerage", "Date": "2025-06-02", "Split Ratio": 2.0},
    ])
    result = deduplicate_split_transactions(df)
    splits = result[result["Type"] == "split"]
    assert len(splits) == 1
    assert splits.iloc[0]["Account"] == "All Accounts"
