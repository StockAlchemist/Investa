# tests/test_portfolio_characterization.py
"""Characterization (golden-master) tests for the portfolio engine.

These pin the CURRENT numeric behaviour of calculate_portfolio_summary and
calculate_historical_performance against a fixed fixture portfolio and a fully
deterministic fake market-data provider (no network, no disk caches). They are
the safety net for splitting portfolio_logic.py: if a refactor changes any
number here, it changed behaviour.

The values were captured on 2026-06-10 from the code as-is. They encode
behaviour, not correctness — if a deliberate calculation fix changes them,
re-capture and update the goldens in the same commit.

Time-anchored metrics (XIRR/MWR, per-holding IRR, day-change) depend on the
wall-clock date and are asserted structurally, not numerically.
"""

import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

# --- Add src directory to sys.path ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

import portfolio_logic as pl
from data_loader import load_and_clean_transactions

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "sample_transactions.csv")
ACC_MAP = {"IBKR": "USD", "SET": "THB"}
HIST_START, HIST_END = date(2023, 1, 1), date(2024, 6, 28)

# --- Deterministic market data ---
PRICES = {"AAPL": 190.0, "MSFT": 350.0, "DELTA.BK": 80.0}
CURRENCY = {"AAPL": "USD", "MSFT": "USD", "DELTA.BK": "THB"}
FX_PER_USD = {"USD": 1.0, "THB": 35.0}


def _bdays(start, end):
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))


class FakeMDP:
    """Stands in for MarketDataProvider with synthetic, deterministic data.

    Historical prices ramp linearly from 60% to 100% of the current price over
    the requested window; FX is constant. Mirrors the provider's contracts:
    quote dicts with 'price'/'currency', frames with a 'price' column.
    """

    def get_current_quotes(self, internal_stock_symbols, required_currencies,
                           user_symbol_map, user_excluded_symbols):
        quotes = {}
        for s in set(internal_stock_symbols):
            if s in PRICES:
                quotes[s] = {
                    "price": PRICES[s],
                    "previous_close": PRICES[s] * 0.99,
                    "prev_close": PRICES[s] * 0.99,
                    "currency": CURRENCY[s],
                    "name": f"{s} Inc",
                }
        fx = {c: FX_PER_USD.get(c, 1.0) for c in set(required_currencies) | {"USD"}}
        return quotes, fx, dict(fx), False, False

    def get_fundamental_data(self, yf_symbol, force_refresh=False):
        return {}

    def get_fundamental_data_batch(self, yf_symbols):
        return {s: {} for s in yf_symbols}

    def _ensure_metadata_batch(self, yf_symbols):
        return {s: {"currency": CURRENCY.get(s, "USD"), "name": s, "quoteType": "EQUITY"}
                for s in yf_symbols}

    def get_historical_data(self, symbols_yf, start_date, end_date, **kwargs):
        idx = _bdays(start_date, end_date)
        out = {}
        for s in symbols_yf:
            if s in PRICES:
                vals = np.linspace(0.6 * PRICES[s], PRICES[s], len(idx))
                out[s] = pd.DataFrame({"price": vals}, index=idx)
        return out, False

    def get_historical_fx_rates(self, fx_pairs_yf, start_date, end_date, **kwargs):
        idx = _bdays(start_date, end_date)
        out = {}
        for p in fx_pairs_yf:
            cur = p.replace("USD", "").replace("=X", "")
            rate = FX_PER_USD.get(cur, 1.0)
            out[p] = pd.DataFrame({"price": [rate] * len(idx)}, index=idx)
        return out, False


@pytest.fixture(scope="module")
def loaded_tx():
    tx, orig, ig_idx, ig_rsn, _, _, _ = load_and_clean_transactions(SAMPLE_CSV, ACC_MAP, "USD")
    assert tx is not None and len(tx) == 9
    return tx, orig, ig_idx, ig_rsn


@pytest.fixture(scope="module")
def summary_result(loaded_tx):
    tx, orig, ig_idx, ig_rsn = loaded_tx
    return pl.calculate_portfolio_summary(
        all_transactions_df_cleaned=tx,
        original_transactions_df_for_ignored=orig,
        ignored_indices_from_load=ig_idx,
        ignored_reasons_from_load=ig_rsn,
        display_currency="USD",
        show_closed_positions=True,
        manual_overrides_dict={},
        user_symbol_map={},
        user_excluded_symbols=set(),
        default_currency="USD",
        account_currency_map=ACC_MAP,
        market_provider=FakeMDP(),
    )


@pytest.fixture(scope="module")
def hist_result(loaded_tx):
    tx, orig, ig_idx, ig_rsn = loaded_tx
    original = pl.get_shared_mdp
    pl.get_shared_mdp = lambda *a, **k: FakeMDP()
    try:
        return pl.calculate_historical_performance(
            all_transactions_df_cleaned=tx,
            original_transactions_df_for_ignored=orig,
            ignored_indices_from_load=ig_idx,
            ignored_reasons_from_load=ig_rsn,
            start_date=HIST_START,
            end_date=HIST_END,
            interval="D",
            benchmark_symbols_yf=[],
            display_currency="USD",
            account_currency_map=ACC_MAP,
            default_currency="USD",
            use_raw_data_cache=False,
            use_daily_results_cache=False,
            user_symbol_map={},
            manual_overrides_dict={},
            user_excluded_symbols=set(),
        )
    finally:
        pl.get_shared_mdp = original


def G(x):
    """Golden value with tight tolerance (floating-point only, no behaviour slack)."""
    return pytest.approx(x, rel=1e-9, abs=1e-9)


# --- calculate_portfolio_summary ---

def test_summary_status_and_shapes(summary_result):
    summary, holdings, _, acct, ig_idx, ig_rsn, status = summary_result
    assert status == "Success (All Accounts)"
    assert isinstance(summary, dict)
    assert isinstance(holdings, pd.DataFrame)
    assert sorted(acct.keys()) == ["IBKR", "SET"]
    assert ig_idx == set() and ig_rsn == {}


def test_summary_overall_metrics_golden(summary_result):
    summary = summary_result[0]
    expected = {
        "market_value": 5021.428571428572,
        "cash_balance": 1142.857142857143,
        "cost_basis_held": 3365.071428571429,
        "unrealized_gain": 1656.357142857143,
        "realized_gain": 117.5,
        "dividends": 5.0,
        "commissions": 19.71428571428571,
        "total_gain": 1759.142857142857,
        "total_buy_cost": 2977.5714285714284,
        "cumulative_investment": 1823.2857142857142,
        "total_cost_invested": 1939.357142857143,
        "fx_gain_loss_display": 1425.7142857142858,
        "total_return_pct": 59.07978697884182,
        "total_dividends_display": 5.0,
        "total_realized_gain_display": 117.5,
    }
    for key, val in expected.items():
        assert summary[key] == G(val), f"summary[{key!r}] drifted"


def test_summary_holdings_golden(summary_result):
    holdings = summary_result[1]
    h = holdings.set_index(["Account", "Symbol"]).sort_index()

    expected = {
        # (Account, Symbol): (Quantity, Avg Cost, Cost Basis, Market Value, Unreal. Gain, Realized Gain, Local Ccy)
        ("IBKR", "AAPL"):     (10.0,    75.25,  752.5,  1900.0, 1147.5, 117.5, "USD"),
        ("IBKR", "MSFT"):     (5.0,     251.0,  1255.0, 1750.0, 495.0,  0.0,   "USD"),
        ("SET", "DELTA.BK"):  (100.0,   2.147142857142857, 214.7142857142857, 228.57142857142856, 13.857142857142861, 0.0, "THB"),
        ("SET", "Cash (฿)"):  (40000.0, 0.02857142857142857, 1142.857142857143, 1142.857142857143, 0.0, 0.0, "THB"),
    }
    assert set(h.index) == set(expected)
    for key, (qty, avg, basis, mv, unreal, realized, ccy) in expected.items():
        row = h.loc[key]
        assert row["Quantity"] == G(qty), f"{key} Quantity"
        assert row["Avg Cost (USD)"] == G(avg), f"{key} Avg Cost"
        assert row["Cost Basis (USD)"] == G(basis), f"{key} Cost Basis"
        assert row["Market Value (USD)"] == G(mv), f"{key} Market Value"
        assert row["Unreal. Gain (USD)"] == G(unreal), f"{key} Unreal. Gain"
        assert row["Realized Gain (USD)"] == G(realized), f"{key} Realized Gain"
        assert row["Local Currency"] == ccy, f"{key} Local Currency"


def test_summary_time_anchored_metrics_are_sane(summary_result):
    """MWR/IRR depend on the wall-clock date — assert structure, not value."""
    summary = summary_result[0]
    assert math.isfinite(summary["portfolio_mwr"])
    assert summary["portfolio_mwr"] > 0  # this fixture portfolio is profitable
    assert isinstance(summary["report_date"], str) or hasattr(summary["report_date"], "isoformat")


# --- calculate_historical_performance ---

def test_historical_status_and_shape(hist_result):
    hist_df, raw_prices, raw_fx, status = hist_result
    assert status.startswith("Success (All Accounts)")
    assert hist_df.shape == (531, 8)
    assert list(hist_df.columns) == [
        "Portfolio Value", "Portfolio Daily Gain", "daily_return",
        "Portfolio Accumulated Gain", "Absolute Gain ($)", "Absolute ROI (%)",
        "Cumulative Net Flow", "drawdown",
    ]
    # First row is the first transaction date; last row is the requested end.
    assert hist_df.index[0].date() == date(2023, 1, 15)
    assert hist_df.index[-1].date() == HIST_END


def test_historical_series_golden(hist_result):
    hist_df = hist_result[0]
    pv = hist_df["Portfolio Value"]
    assert pv.iloc[0] == G(2319.0745501285346)
    assert pv.iloc[260] == G(4241.422695556372)
    assert pv.iloc[-1] == G(5015.142857142857)

    assert hist_df["Portfolio Accumulated Gain"].iloc[-1] == G(1.457530416842845)  # TWR factor
    assert hist_df["Cumulative Net Flow"].iloc[-1] == G(1138.5714285714287)
    assert hist_df["Absolute Gain ($)"].iloc[-1] == G(3876.5714285714284)
    assert hist_df["drawdown"].min() == G(-25.037449835855707)


def test_historical_internal_consistency(hist_result):
    """Invariants that must hold regardless of golden values."""
    hist_df = hist_result[0]
    pv = hist_df["Portfolio Value"]
    assert (pv > 0).all()
    # daily_return is NaN on the first day (no prior close) and defined after.
    assert not hist_df["daily_return"].iloc[1:].isna().any()
    assert hist_df["drawdown"].max() <= 0 + 1e-12
    # Absolute Gain = Portfolio Value - Cumulative Net Flow (by construction)
    np.testing.assert_allclose(
        hist_df["Absolute Gain ($)"],
        pv - hist_df["Cumulative Net Flow"],
        rtol=1e-9,
    )
