"""The FIFO cache is keyed by what the lots are computed from, not by a file mtime."""

import os
import sys

import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from portfolio_logic import _transactions_fingerprint  # noqa: E402


def _tx(currency="USD"):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-02", "2025-03-04"]),
            "Symbol": ["AAPL", "AAPL"],
            "Quantity": [10.0, -4.0],
            "Local Currency": [currency, currency],
        }
    )


def test_same_transactions_share_a_key():
    # Two users holding identical books may share the result; it is identical.
    assert _transactions_fingerprint(_tx()) == _transactions_fingerprint(_tx())


def test_a_change_the_db_mtime_never_sees_changes_the_key():
    # An account-currency map fills blank currencies at load time, so the
    # transactions change while the DB file (and its mtime) stays put.
    assert _transactions_fingerprint(_tx("USD")) != _transactions_fingerprint(
        _tx("THB")
    )


def test_order_and_columns_are_part_of_the_key():
    df = _tx()
    assert _transactions_fingerprint(df) != _transactions_fingerprint(df.iloc[::-1])
    renamed = df.rename(columns={"Quantity": "Qty"})
    assert _transactions_fingerprint(df) != _transactions_fingerprint(renamed)
