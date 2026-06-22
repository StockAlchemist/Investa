"""Regression tests for the AI chat portfolio-summary tool.

Pins the fix for the currency bug: the tool must report per-holding values in the
display currency (USD) sourced from summary_df — not None values read off the raw
native-currency holdings_dict, which made the model reconstruct figures from
unconverted native prices and rank a THB fund as the largest position.
"""

import numpy as np
import pandas as pd
import pytest

import server.ai_chat_service as svc


class _FakeConfigManager:
    manual_overrides: dict = {}

    def load_manual_overrides(self):
        pass


@pytest.fixture
def patched_tool(monkeypatch):
    # summary_df mirrors the real per-holding columns (already in USD).
    summary_df = pd.DataFrame([
        {"Symbol": "GOOG", "Account": "IBKR", "Quantity": 1314.0,
         "Market Value (USD)": 482842.43, "Total Gain (USD)": 284036.0, "Total Return %": 142.54},
        # A Thai fund: large native (THB) NAV, but small once converted to USD.
        {"Symbol": "SCBRMS&P500", "Account": "SET", "Quantity": 88177.0,
         "Market Value (USD)": 60715.93, "Total Gain (USD)": 5000.0, "Total Return %": 9.0},
        # Aggregate row must be excluded.
        {"Symbol": "Total", "Account": "", "Quantity": np.nan,
         "Market Value (USD)": 999999.0, "Total Gain (USD)": np.nan, "Total Return %": np.nan},
        # Closed/zero-qty position must be skipped.
        {"Symbol": "ZERO", "Account": "IBKR", "Quantity": 0.0,
         "Market Value (USD)": 100.0, "Total Gain (USD)": 0.0, "Total Return %": 0.0},
        # Missing price -> market_value should serialize as None, not crash.
        {"Symbol": "NANMV", "Account": "IBKR", "Quantity": 5.0,
         "Market Value (USD)": np.nan, "Total Gain (USD)": np.nan, "Total Return %": np.nan},
    ])
    overall = {"market_value": 1838470.51, "total_gain": 1791398.92,
               "total_return_pct": 40.82, "cash_balance": 19151.39}

    fake_data = (pd.DataFrame({"x": [1]}), {}, {}, set(), {}, {}, "/db", 0.0)
    monkeypatch.setattr(svc, "get_transaction_data", lambda u: fake_data)
    monkeypatch.setattr(svc, "get_config_manager", lambda u: _FakeConfigManager())
    monkeypatch.setattr(svc, "get_shared_mdp", lambda: object())
    monkeypatch.setattr(
        svc, "calculate_portfolio_summary",
        lambda **kw: (overall, summary_df, {}, {}, None, None, "ok"),
    )
    return svc.get_portfolio_summary_tool(object())


def test_holdings_have_usd_values_not_none(patched_tool):
    by_symbol = {h["symbol"]: h for h in patched_tool["holdings"]}
    assert by_symbol["GOOG"]["market_value"] == 482842.43
    assert by_symbol["GOOG"]["profit"] == 284036.0
    assert by_symbol["GOOG"]["return_pct"] == 142.54
    # Every holding must declare its currency, and the tool labels the payload USD.
    assert patched_tool["currency"] == "USD"
    assert all(h["currency"] == "USD" for h in patched_tool["holdings"])


def test_thai_fund_value_is_usd_not_native(patched_tool):
    fund = next(h for h in patched_tool["holdings"] if h["symbol"] == "SCBRMS&P500")
    # The USD value, not an unconverted THB figure in the millions.
    assert fund["market_value"] == 60715.93


def test_sorted_by_market_value_and_filtered(patched_tool):
    symbols = [h["symbol"] for h in patched_tool["holdings"]]
    assert symbols[0] == "GOOG"  # largest USD position first
    assert "Total" not in symbols  # aggregate row excluded
    assert "ZERO" not in symbols   # zero-qty position skipped
    # GOOG outranks the Thai fund despite the fund's larger native quantity.
    assert symbols.index("GOOG") < symbols.index("SCBRMS&P500")


def test_missing_market_value_serializes_as_none(patched_tool):
    nanmv = next(h for h in patched_tool["holdings"] if h["symbol"] == "NANMV")
    assert nanmv["market_value"] is None
    assert nanmv["qty"] == 5.0  # still included; only the price is unknown
