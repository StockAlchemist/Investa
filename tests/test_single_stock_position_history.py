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
