"""Auto-add-cash leg generation for Manual-mode accounts.

The principal leg must carry the *gross* trade amount and the commission must
appear exactly once, as its own leg. The regression these tests guard is a
commission double-charge: the web/SwiftUI forms and ibkr_connector all fold
commission into "Total Amount", so deriving the principal from that field
charged it twice (once inside the principal, once as the fee leg).
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import server.routes.transactions as tx_routes
from server.routes.transactions import _handle_auto_cash_generation


@pytest.fixture
def captured(monkeypatch):
    """Capture generated legs instead of writing them to a database.

    monkeypatch (not a module-level rebind) so the stub is reverted after each
    test — the previous version leaked a patched add_transaction_to_db into
    every test collected afterwards.
    """
    rows = []

    def fake_add(conn, tx_data):
        rows.append(tx_data)
        return True, len(rows)

    monkeypatch.setattr(tx_routes, "add_transaction_to_db", fake_add)
    return rows


def _trade(**overrides):
    """A buy of 10 @ 150 with a 5.00 commission, as the clients send it:
    Total Amount = qty * price + commission = 1505.00, stored signed."""
    tx = {
        "Date": "2026-03-13",
        "Type": "Buy",
        "Symbol": "AAPL",
        "Quantity": 10.0,
        "Price/Share": 150.0,
        "Commission": 5.0,
        "Total Amount": -1505.0,
        "Account": "TestAcc",
        "Local Currency": "USD",
        "Auto-add Cash": True,
    }
    tx.update(overrides)
    return tx


def test_buy_principal_excludes_commission(captured):
    """Cash out must total 1505 (1500 principal + 5 fee), not 1510."""
    _handle_auto_cash_generation(None, _trade())

    assert len(captured) == 2
    principal, fee = captured

    assert principal["Type"] == "Sell"
    assert principal["Symbol"] == "$CASH"
    assert principal["Quantity"] == pytest.approx(1500.0)
    assert principal["Total Amount"] == pytest.approx(1500.0)
    assert principal["Note"] == "Auto-cash for Buy AAPL"

    assert fee["Type"] == "Withdrawal"
    assert fee["Quantity"] == pytest.approx(5.0)
    assert fee["Note"] == "Auto-cash Fee for Buy AAPL"

    # The whole point: commission appears once, not folded into the principal.
    assert principal["Quantity"] + fee["Quantity"] == pytest.approx(1505.0)


def test_sell_principal_excludes_commission(captured):
    """Proceeds must net to 1990 (2000 principal - 10 fee), not 1980."""
    _handle_auto_cash_generation(
        None,
        _trade(
            Type="Sell",
            Symbol="MSFT",
            Quantity=5.0,
            **{
                "Price/Share": 400.0,
                "Commission": 10.0,
                "Total Amount": 1990.0,
            },
        ),
    )

    assert len(captured) == 2
    principal, fee = captured

    assert principal["Type"] == "Buy"
    assert principal["Quantity"] == pytest.approx(2000.0)
    assert principal["Note"] == "Auto-cash for Sell MSFT"
    assert fee["Quantity"] == pytest.approx(10.0)
    assert fee["Note"] == "Auto-cash Fee for Sell MSFT"

    assert principal["Quantity"] - fee["Quantity"] == pytest.approx(1990.0)


def test_legacy_total_amount_convention_still_correct(captured):
    """Rows storing Total Amount *net* of commission (the convention in the
    existing corpus) must be unaffected — the principal comes from qty * price
    either way, so both conventions converge on the same legs."""
    _handle_auto_cash_generation(None, _trade(**{"Total Amount": -1500.0}))

    principal, fee = captured
    assert principal["Quantity"] == pytest.approx(1500.0)
    assert fee["Quantity"] == pytest.approx(5.0)


def test_zero_commission_posts_only_principal(captured):
    _handle_auto_cash_generation(
        None, _trade(Commission=0.0, **{"Total Amount": -1500.0})
    )

    assert len(captured) == 1
    assert captured[0]["Quantity"] == pytest.approx(1500.0)


def test_missing_price_falls_back_to_total_amount(captured):
    """Batch/PDF imports default Price/Share to 0. The settlement amount then
    carries the commission, so it gets backed out by direction."""
    _handle_auto_cash_generation(
        None, _trade(**{"Price/Share": 0.0, "Total Amount": -1505.0})
    )

    principal, fee = captured
    assert principal["Quantity"] == pytest.approx(1500.0)
    assert fee["Quantity"] == pytest.approx(5.0)


def test_auto_mode_account_generates_nothing(captured):
    """Auto-mode accounts have the engine derive cash from the trade row, so
    explicit legs would double-count every trade."""
    _handle_auto_cash_generation(None, _trade(), {"TestAcc": "Auto"})

    assert captured == []


def test_unmapped_account_defaults_to_manual(captured):
    _handle_auto_cash_generation(None, _trade(), {"SomeOtherAcc": "Auto"})

    assert len(captured) == 2


def test_total_amount_sign_convention(captured):
    """Buy/Withdrawal legs store a negative Total Amount, Sell positive —
    matching what the clients write and what the existing corpus contains."""
    _handle_auto_cash_generation(None, _trade())
    assert captured[0]["Total Amount"] > 0  # Sell $CASH
    assert captured[1]["Total Amount"] < 0  # Withdrawal $CASH

    captured.clear()
    _handle_auto_cash_generation(
        None,
        _trade(Type="Sell", **{"Total Amount": 1495.0}),
    )
    assert captured[0]["Total Amount"] < 0  # Buy $CASH
    assert captured[1]["Total Amount"] < 0  # Withdrawal $CASH


def test_cash_symbol_and_non_trade_types_are_ignored(captured):
    _handle_auto_cash_generation(None, _trade(Symbol="$CASH"))
    _handle_auto_cash_generation(None, _trade(Type="Dividend"))

    assert captured == []
