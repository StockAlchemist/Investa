"""Unit tests for the single stock position history endpoint."""

import os
import sys
import asyncio
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from server.routes.portfolio import get_stock_position_history
from server.auth import User


def test_single_stock_position_history_calculation(monkeypatch):
    """Test get_stock_position_history calculates value, cost basis, unrealized gain, and return % correctly."""

    async def _test():
        txs = [
            # Lot 1: Buy 100 shares of AAPL @ $150 on 2024-01-10
            {
                "Date": pd.Timestamp("2024-01-10"),
                "Symbol": "AAPL",
                "Account": "IBKR",
                "Type": "Buy",
                "Quantity": 100.0,
                "Price/Share": 150.0,
                "Commission": 0.0,
                "Split Ratio": None,
                "Local Currency": "USD",
                "original_index": 1,
            },
            # Lot 2: Buy 50 shares of AAPL @ $180 on 2024-06-15
            {
                "Date": pd.Timestamp("2024-06-15"),
                "Symbol": "AAPL",
                "Account": "IBKR",
                "Type": "Buy",
                "Quantity": 50.0,
                "Price/Share": 180.0,
                "Commission": 0.0,
                "Split Ratio": None,
                "Local Currency": "USD",
                "original_index": 2,
            },
        ]
        df_txs = pd.DataFrame(txs)

        mock_user = User(id=1, username="testuser", created_at="2024-01-01T00:00:00Z")
        data_tuple = (
            df_txs,  # df
            {},  # manual_overrides
            {},  # user_symbol_map
            set(),  # user_excluded_symbols
            {"IBKR": "USD"},  # account_currency_map
            {},  # account_cash_mode_map
            None,  # original_csv_path
            12345.0,  # db_mtime
        )

        # Mock get_mdp to return synthetic prices
        dates = pd.date_range("2024-01-08", "2024-06-20", freq="B", tz="UTC")
        prices = [150.0 + i * 1.0 for i in range(len(dates))]
        mock_stock_df = pd.DataFrame({"price": prices}, index=dates)

        class MockMDP:
            def get_historical_data(self, symbols, start_date, end_date, interval="1d"):
                return {"AAPL": mock_stock_df}, {}

        monkeypatch.setattr("server.routes.portfolio.get_mdp", lambda: MockMDP())

        res = await get_stock_position_history(
            symbol="AAPL",
            currency="USD",
            period="all",
            data=data_tuple,
            current_user=mock_user,
        )

        assert isinstance(res, list)
        assert len(res) > 0

        # Points exist
        first_pt = res[0]
        assert "date" in first_pt
        assert "value" in first_pt
        assert "cost_basis" in first_pt
        assert "shares" in first_pt
        assert "return_pct" in first_pt

        # Find point on 2024-01-10
        pt_jan10 = next((p for p in res if p["date"] == "2024-01-10"), None)
        assert pt_jan10 is not None
        assert pt_jan10["shares"] == 100.0
        assert pt_jan10["cost_basis"] == 15000.0

        # Find point after second buy (e.g. 2024-06-17)
        pt_after = next((p for p in res if p["date"] >= "2024-06-17"), None)
        assert pt_after is not None
        assert pt_after["shares"] == 150.0
        assert pt_after["cost_basis"] == 100.0 * 150.0 + 50.0 * 180.0  # 24000.0
        assert pt_after["value"] > 0
        assert pt_after["unrealized_gain"] == pytest.approx(
            pt_after["value"] - pt_after["cost_basis"], 0.01
        )

    asyncio.run(_test())


def test_single_stock_position_history_split_unadjustment(monkeypatch):
    """Verify that split transactions un-adjust historical prices so market value doesn't spike artificially."""

    async def _test():
        txs = [
            # Buy 100 shares of AAPL @ $100 on 2014-01-10
            {
                "Date": pd.Timestamp("2014-01-10"),
                "Symbol": "AAPL",
                "Account": "IBKR",
                "Type": "Buy",
                "Quantity": 100.0,
                "Price/Share": 100.0,
                "Commission": 0.0,
                "Split Ratio": None,
                "Local Currency": "USD",
                "original_index": 1,
            },
            # 7:1 Stock Split on 2014-06-09
            {
                "Date": pd.Timestamp("2014-06-09"),
                "Symbol": "AAPL",
                "Account": "All Accounts",
                "Type": "Split",
                "Quantity": 0.0,
                "Price/Share": 0.0,
                "Commission": 0.0,
                "Split Ratio": 7.0,
                "Local Currency": "USD",
                "original_index": 2,
            },
        ]
        df_txs = pd.DataFrame(txs)

        mock_user = User(id=1, username="testuser", created_at="2014-01-01T00:00:00Z")
        data_tuple = (
            df_txs,
            {},
            {},
            set(),
            {"IBKR": "USD"},
            {},
            None,
            12345.0,
        )

        # In yfinance, prices are split-adjusted (e.g. $14.28 before split, $14.28 after split)
        dates = pd.date_range("2014-06-05", "2014-06-12", freq="B", tz="UTC")
        prices = [14.28] * len(dates)  # Flat split-adjusted price
        mock_stock_df = pd.DataFrame({"price": prices}, index=dates)

        class MockMDP:
            def get_historical_data(self, symbols, start_date, end_date, interval="1d"):
                return {"AAPL": mock_stock_df}, {}

        monkeypatch.setattr("server.routes.portfolio.get_mdp", lambda: MockMDP())

        res = await get_stock_position_history(
            symbol="AAPL",
            currency="USD",
            period="all",
            data=data_tuple,
            current_user=mock_user,
        )

        # Before split (2014-06-06): 100 shares @ ($14.28 * 7 = $100.0) -> value = ~10000.0
        pt_pre = next((p for p in res if p["date"] == "2014-06-06"), None)
        assert pt_pre is not None
        assert pt_pre["shares"] == 100.0
        assert pt_pre["value"] == pytest.approx(10000.0, 50.0)

        # After split (2014-06-09): 700 shares @ $14.28 -> value = ~10000.0
        pt_post = next((p for p in res if p["date"] == "2014-06-09"), None)
        assert pt_post is not None
        assert pt_post["shares"] == 700.0
        assert pt_post["value"] == pytest.approx(10000.0, 50.0)

    asyncio.run(_test())


def test_single_stock_position_history_display_currency_conversion(monkeypatch):
    """A USD position rendered in THB converts value and cost basis at the daily FX rate.

    The endpoint pulls only its own currency pair (it used to recompute the whole
    portfolio's daily history just to read this one series), so the FX fetch is
    asserted here as well.
    """

    async def _test():
        txs = [
            {
                "Date": pd.Timestamp("2024-01-10"),
                "Symbol": "AAPL",
                "Account": "IBKR",
                "Type": "Buy",
                "Quantity": 100.0,
                "Price/Share": 150.0,
                "Commission": 0.0,
                "Split Ratio": None,
                "Local Currency": "USD",
                "original_index": 1,
            },
        ]
        df_txs = pd.DataFrame(txs)

        mock_user = User(id=1, username="testuser", created_at="2024-01-01T00:00:00Z")
        data_tuple = (
            df_txs,
            {},
            {},
            set(),
            {"IBKR": "USD"},
            {},
            None,
            12345.0,
        )

        dates = pd.date_range("2024-01-08", "2024-03-01", freq="B", tz="UTC")
        mock_stock_df = pd.DataFrame({"price": [150.0] * len(dates)}, index=dates)
        mock_fx_df = pd.DataFrame({"price": [33.0] * len(dates)}, index=dates)
        fx_pairs_requested = []

        class MockMDP:
            def get_historical_data(self, symbols, start_date, end_date, interval="1d"):
                return {"AAPL": mock_stock_df}, {}

            def get_historical_fx_rates(
                self,
                fx_pairs_yf,
                start_date,
                end_date,
                interval="1d",
                use_cache=True,
                cache_key=None,
                cache_file=None,
            ):
                fx_pairs_requested.extend(fx_pairs_yf)
                return {fx_pairs_yf[0]: mock_fx_df}, False

        monkeypatch.setattr("server.routes.portfolio.get_mdp", lambda: MockMDP())

        res = await get_stock_position_history(
            symbol="AAPL",
            currency="THB",
            period="all",
            data=data_tuple,
            current_user=mock_user,
        )

        # Yahoo quotes THB as USDTHB=X (THB per USD), not THB=X.
        assert fx_pairs_requested == ["USDTHB=X"]

        pt = next((p for p in res if p["date"] >= "2024-01-11"), None)
        assert pt is not None
        assert pt["shares"] == 100.0
        assert pt["value"] == pytest.approx(100.0 * 150.0 * 33.0, rel=1e-6)
        assert pt["cost_basis"] == pytest.approx(100.0 * 150.0 * 33.0, rel=1e-6)

    asyncio.run(_test())


def test_single_stock_position_history_local_currency_converted_to_usd(monkeypatch):
    """A THB-denominated position rendered in USD is divided by the daily USDTHB rate.

    The local currency used to be ignored unless it was USD, so a SET holding
    showed its raw baht under a "$" label.
    """

    async def _test():
        txs = [
            {
                "Date": pd.Timestamp("2024-01-10"),
                "Symbol": "AOT:BKK",
                "Account": "Kim Eng",
                "Type": "Buy",
                "Quantity": 1000.0,
                "Price/Share": 60.0,
                "Commission": 0.0,
                "Split Ratio": None,
                "Local Currency": "THB",
                "original_index": 1,
            },
        ]
        df_txs = pd.DataFrame(txs)

        mock_user = User(id=1, username="testuser", created_at="2024-01-01T00:00:00Z")
        data_tuple = (
            df_txs,
            {},
            {},
            set(),
            {"Kim Eng": "THB"},
            {},
            None,
            12345.0,
        )

        dates = pd.date_range("2024-01-08", "2024-03-01", freq="B", tz="UTC")
        mock_stock_df = pd.DataFrame({"price": [60.0] * len(dates)}, index=dates)
        mock_fx_df = pd.DataFrame({"price": [30.0] * len(dates)}, index=dates)

        class MockMDP:
            def get_historical_data(self, symbols, start_date, end_date, interval="1d"):
                # Keyed by the mapped Yahoo ticker the route actually requested.
                return {symbols[0]: mock_stock_df}, {}

            def get_historical_fx_rates(
                self,
                fx_pairs_yf,
                start_date,
                end_date,
                interval="1d",
                use_cache=True,
                cache_key=None,
                cache_file=None,
            ):
                return {p: mock_fx_df for p in fx_pairs_yf}, False

        monkeypatch.setattr("server.routes.portfolio.get_mdp", lambda: MockMDP())

        res = await get_stock_position_history(
            symbol="AOT:BKK",
            currency="USD",
            period="all",
            data=data_tuple,
            current_user=mock_user,
        )

        pt = next((p for p in res if p["date"] >= "2024-01-11"), None)
        assert pt is not None
        assert pt["shares"] == 1000.0
        # 1000 shares * ฿60 = ฿60,000 -> $2,000 at 30 THB/USD
        assert pt["value"] == pytest.approx(2000.0, rel=1e-6)
        assert pt["cost_basis"] == pytest.approx(2000.0, rel=1e-6)

    asyncio.run(_test())
