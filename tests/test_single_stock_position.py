"""Unit tests for the single stock position and return tracking endpoint."""

import os
import sys
from datetime import date
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from portfolio_analyzer import calculate_fifo_lots_and_gains


def test_calculate_fifo_lots_and_gains_single_stock():
    """Verify FIFO lots, splits, and realized sales for a single stock."""
    txs = [
        # Lot 1: Buy 100 shares of AAPL @ $150 on 2024-01-10
        {
            "Date": pd.Timestamp("2024-01-10"),
            "Symbol": "AAPL",
            "Account": "IBKR",
            "Type": "Buy",
            "Quantity": 100.0,
            "Price/Share": 150.0,
            "Commission": 5.0,
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
            "Commission": 2.5,
            "Local Currency": "USD",
            "original_index": 2,
        },
        # Partial Sell: Sell 60 shares of AAPL @ $200 on 2024-11-20 (consumes 60 from Lot 1)
        {
            "Date": pd.Timestamp("2024-11-20"),
            "Symbol": "AAPL",
            "Account": "IBKR",
            "Type": "Sell",
            "Quantity": 60.0,
            "Price/Share": 200.0,
            "Commission": 3.0,
            "Local Currency": "USD",
            "original_index": 3,
        },
    ]

    df_txs = pd.DataFrame(txs)
    hist_fx = {}

    df_gains, open_lots = calculate_fifo_lots_and_gains(
        transactions_df=df_txs,
        display_currency="USD",
        historical_fx_yf=hist_fx,
        default_currency="USD",
        shortable_symbols=set(),
    )

    # 1. Realized Gains verification
    assert len(df_gains) == 1
    gain_row = df_gains.iloc[0]
    assert gain_row["Symbol"] == "AAPL"
    assert gain_row["Quantity"] == 60.0
    # Proceeds = 60 * 200 - 3 commission = 11997.0
    assert pytest.approx(gain_row["Total Proceeds (Local)"], 0.01) == 11997.0
    # Cost basis for 60 shares of Lot 1 (cost per share = 150 + 5/100 = 150.05) = 60 * 150.05 = 9003.0
    assert pytest.approx(gain_row["Total Cost Basis (Local)"], 0.01) == 9003.0
    # Realized Gain = 11997 - 9003 = 2994.0
    assert pytest.approx(gain_row["Realized Gain (Local)"], 0.01) == 2994.0

    # 2. Open Lots verification
    aapl_lots = open_lots.get(("AAPL", "IBKR"), [])
    assert len(aapl_lots) == 2

    # Remaining Lot 1: 40 shares @ $150.05
    lot1 = aapl_lots[0]
    assert pytest.approx(lot1["qty"]) == 40.0
    assert pytest.approx(lot1["cost_per_share_local_net"], 0.01) == 150.05
    assert lot1["purchase_date"] == date(2024, 1, 10)

    # Lot 2: 50 shares @ $180.05 (180 + 2.5/50)
    lot2 = aapl_lots[1]
    assert pytest.approx(lot2["qty"]) == 50.0
    assert pytest.approx(lot2["cost_per_share_local_net"], 0.01) == 180.05
    assert lot2["purchase_date"] == date(2024, 6, 15)

    # Total remaining open shares = 90
    total_open_qty = sum(lot["qty"] for lot in aapl_lots)
    assert pytest.approx(total_open_qty) == 90.0


def test_calculate_fifo_lots_with_stock_split():
    """Verify FIFO lots adjust quantity and unit cost proportionally on split."""
    txs = [
        # Buy 10 shares of NVDA @ $600 on 2024-01-10
        {
            "Date": pd.Timestamp("2024-01-10"),
            "Symbol": "NVDA",
            "Account": "Main",
            "Type": "Buy",
            "Quantity": 10.0,
            "Price/Share": 600.0,
            "Commission": 0.0,
            "Local Currency": "USD",
            "original_index": 1,
        },
        # 10-for-1 Split on 2024-06-10 (Split Ratio = 10.0)
        {
            "Date": pd.Timestamp("2024-06-10"),
            "Symbol": "NVDA",
            "Account": "Main",
            "Type": "Stock Split",
            "Split Ratio": 10.0,
            "original_index": 2,
        },
    ]

    df_txs = pd.DataFrame(txs)
    df_gains, open_lots = calculate_fifo_lots_and_gains(
        transactions_df=df_txs,
        display_currency="USD",
        historical_fx_yf={},
        default_currency="USD",
        shortable_symbols=set(),
    )

    nvda_lots = open_lots.get(("NVDA", "MAIN"), [])
    assert len(nvda_lots) == 1
    assert pytest.approx(nvda_lots[0]["qty"]) == 100.0
    assert pytest.approx(nvda_lots[0]["cost_per_share_local_net"]) == 60.0


def test_calculate_fifo_lots_with_account_transfer():
    """Verify FIFO lots preserve original purchase date and unit cost across account transfers."""
    txs = [
        # Buy 50 shares of TSLA @ $200 in Account A
        {
            "Date": pd.Timestamp("2023-05-10"),
            "Symbol": "TSLA",
            "Account": "AccountA",
            "Type": "Buy",
            "Quantity": 50.0,
            "Price/Share": 200.0,
            "Commission": 0.0,
            "Local Currency": "USD",
            "original_index": 1,
        },
        # Transfer 30 shares from Account A to Account B on 2024-02-01
        {
            "Date": pd.Timestamp("2024-02-01"),
            "Symbol": "TSLA",
            "Account": "AccountA",
            "To Account": "AccountB",
            "Type": "Transfer",
            "Quantity": 30.0,
            "Price/Share": 200.0,
            "original_index": 2,
        },
    ]

    df_txs = pd.DataFrame(txs)
    df_gains, open_lots = calculate_fifo_lots_and_gains(
        transactions_df=df_txs,
        display_currency="USD",
        historical_fx_yf={},
        default_currency="USD",
        shortable_symbols=set(),
    )

    lots_a = open_lots.get(("TSLA", "ACCOUNTA"), [])
    lots_b = open_lots.get(("TSLA", "ACCOUNTB"), [])

    assert len(lots_a) == 1
    assert pytest.approx(lots_a[0]["qty"]) == 20.0
    assert pytest.approx(lots_a[0]["cost_per_share_local_net"]) == 200.0
    assert lots_a[0]["purchase_date"] == date(2023, 5, 10)

    assert len(lots_b) == 1
    assert pytest.approx(lots_b[0]["qty"]) == 30.0
    assert pytest.approx(lots_b[0]["cost_per_share_local_net"]) == 200.0
    assert lots_b[0]["purchase_date"] == date(2023, 5, 10)  # Preserved!


def test_get_stock_position_endpoint():
    """Verify GET /api/stock/{symbol}/position returns 200 without crashes."""
    from fastapi.testclient import TestClient
    from server.main import app
    from server.dependencies import get_current_user
    from server.auth import User

    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, username="test", created_at="2024-01-01"
    )
    client = TestClient(app)
    try:
        # Test for an unheld stock (e.g. AMZN)
        resp = client.get("/api/stock/AMZN/position?currency=USD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AMZN"
        assert data["display_currency"] == "USD"
        assert "has_position" in data
        assert "open_lots" in data
        assert "closed_trades" in data
    finally:
        app.dependency_overrides.clear()



