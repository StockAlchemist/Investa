"""The performance graph and the summary must value the portfolio identically.

The graph comes from the historical engine and the headline total from the
summary engine. Where the two engines answered differently the chart drifted
away from the number in the hero panel, so these pin the two rules that were
out of step: manually priced holdings, and income rows booked on $CASH.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from portfolio_history import manual_prices_by_symbol_id  # noqa: E402
from portfolio_valuation_kernels import (  # noqa: E402
    _calculate_daily_holdings_chronological_numba,
    _manual_override_price,
)


def _resolve_cash_modes(account_cash_mode_map, account_to_id):
    """The cash modes the real builder hands the kernel for these settings.

    The settings file keeps the case the user typed and the engine's account ids
    may or may not be normalized; a mismatch used to mean a silent Manual.
    """
    import portfolio_valuation_kernels as k

    captured = {}
    real = k._calculate_holdings_numba

    def spy(*args, **kwargs):
        captured["modes"] = args[-1]
        return real(*args, **kwargs)

    k._calculate_holdings_numba = spy
    try:
        k._calculate_portfolio_value_at_date_unadjusted_numba(
            target_date=pd.Timestamp("2026-08-10").date(),
            transactions_df=_empty_tx(account_to_id),
            historical_prices_yf_unadjusted={},
            historical_fx_yf={},
            target_currency="USD",
            internal_to_yf_map={},
            account_currency_map={a: "USD" for a in account_to_id},
            default_currency="USD",
            manual_overrides_dict=None,
            processed_warnings=set(),
            symbol_to_id={"$CASH": 0},
            id_to_symbol={0: "$CASH"},
            account_to_id=account_to_id,
            id_to_account={i: a for a, i in account_to_id.items()},
            type_to_id={"deposit": 0},
            currency_to_id={"USD": 0},
            id_to_currency={0: "USD"},
            account_cash_mode_map=account_cash_mode_map,
        )
    finally:
        k._calculate_holdings_numba = real
    return list(captured["modes"])


def _empty_tx(account_to_id):
    account = next(iter(account_to_id))
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-01-02", tz="UTC"),
                "Type": "deposit",
                "Symbol": "$CASH",
                "Quantity": 0.0,
                "Price/Share": 1.0,
                "Total Amount": 0.0,
                "Commission": 0.0,
                "Account": account,
                "Split Ratio": np.nan,
                "Note": "",
                "Local Currency": "USD",
                "To Account": np.nan,
                "original_index": 0,
            }
        ]
    )


def test_cash_mode_resolves_regardless_of_the_case_the_user_typed():
    # Engine-side ids arrive uppercased from generate_mappings; settings do not.
    account_to_id = {"IBKR DHEE": 0, "MORGAN STANLEY": 1, "SCBAM": 2, "WEBULL": 3}
    settings = {
        "IBKR Dhee": "Auto",
        "Morgan Stanley": "Auto",
        "SCBAM": "Auto",
        "Webull": "auto",  # tolerate a lowercased mode too
    }
    assert _resolve_cash_modes(settings, account_to_id) == [1, 1, 1, 1]


def test_cash_mode_resolves_when_ids_keep_the_original_case():
    account_to_id = {"IBKR Dhee": 0, "Webull": 1}
    assert _resolve_cash_modes({"IBKR Dhee": "Auto", "Webull": "Auto"}, account_to_id)


def test_manual_and_unknown_accounts_stay_manual():
    account_to_id = {"IBKR DHEE": 0, "SET": 1}
    settings = {"IBKR Dhee": "Manual", "Retired Account": "Auto"}
    assert _resolve_cash_modes(settings, account_to_id) == [0, 0]
    assert _resolve_cash_modes(None, account_to_id) == [0, 0]


# --- Manual price overrides (holdings with no market feed) ---


def test_manual_override_price_reads_the_declared_nav():
    overrides = {"SCBRMS&P500": {"price": 22.4124, "asset_type": "ETF"}}
    assert _manual_override_price(overrides, "SCBRMS&P500") == 22.4124
    # The engine normalizes symbols; the override file keeps the user's casing.
    assert _manual_override_price(overrides, "scbrms&p500") == 22.4124
    assert _manual_override_price(overrides, "AAPL") is None


def test_manual_override_price_rejects_unusable_values():
    for bad in ({"price": 0}, {"price": -3}, {"price": None}, {"price": "n/a"}, {}):
        assert _manual_override_price({"FUND": bad}, "FUND") is None
    assert _manual_override_price(None, "FUND") is None


def test_manual_prices_resolve_to_symbol_ids():
    overrides = {"es-gqg": {"price": 24.1907}, "NOPRICE": {"asset_type": "ETF"}}
    id_to_symbol = {0: "ES-GQG", 1: "AAPL", 2: "NOPRICE"}
    assert manual_prices_by_symbol_id(overrides, id_to_symbol) == {0: 24.1907}
    assert manual_prices_by_symbol_id({}, id_to_symbol) == {}


# --- $CASH income rows in Manual cash mode ---

# Type ids used by the chronological kernel, chosen arbitrarily but distinctly.
BUY, DEPOSIT, SELL, WITHDRAWAL = 1, 2, 3, 4
DIVIDEND, INTEREST, FEES, TAX = 5, 6, 7, 8
TRANSFER, SPLIT, STOCK_SPLIT, SHORT_SELL, BUY_TO_COVER, SPIN_OFF = 9, 10, 11, 12, 13, 14
CASH_SYMBOL_ID = 0


def _cash_balance(rows, cash_mode, commission=0.0, totals=None, prices=None):
    """Run the chronological kernel over `rows` and return the closing cash.

    Each row is (type_id, quantity) booked on $CASH for the single account;
    `totals` and `prices` override the Total Amount / Price columns so the
    per-column conventions can be pinned independently.
    """
    n = len(rows)
    day = pd.Timestamp("2026-08-10").toordinal()
    _, cash, _ = _calculate_daily_holdings_chronological_numba(
        date_ordinals_np=np.array([day], dtype=np.int64),
        tx_dates_ordinal_np=np.full(n, day, dtype=np.int64),
        tx_symbols_np=np.zeros(n, dtype=np.int64),
        tx_accounts_np=np.zeros(n, dtype=np.int64),
        tx_types_np=np.array([r[0] for r in rows], dtype=np.int64),
        tx_quantities_np=np.array([r[1] for r in rows], dtype=np.float64),
        tx_prices_np=(
            np.ones(n, dtype=np.float64)
            if prices is None
            else np.array(prices, dtype=np.float64)
        ),
        tx_commissions_np=np.full(n, commission, dtype=np.float64),
        tx_split_ratios_np=np.zeros(n, dtype=np.float64),
        tx_to_accounts_np=np.full(n, -1, dtype=np.int64),
        tx_totals_np=np.array(
            [r[1] for r in rows] if totals is None else totals, dtype=np.float64
        ),
        num_symbols=1,
        num_accounts=1,
        split_type_id=SPLIT,
        stock_split_type_id=STOCK_SPLIT,
        buy_type_id=BUY,
        deposit_type_id=DEPOSIT,
        sell_type_id=SELL,
        withdrawal_type_id=WITHDRAWAL,
        short_sell_type_id=SHORT_SELL,
        buy_to_cover_type_id=BUY_TO_COVER,
        transfer_type_id=TRANSFER,
        spin_off_type_id=SPIN_OFF,
        fees_type_id=FEES,
        dividend_type_id=DIVIDEND,
        interest_type_id=INTEREST,
        tax_type_id=TAX,
        cash_symbol_id=CASH_SYMBOL_ID,
        stock_qty_close_tolerance=1e-9,
        shortable_symbol_ids=np.zeros(0, dtype=np.int64),
        acc_cash_modes=np.array([1 if cash_mode == "auto" else 0], dtype=np.int64),
    )
    return cash[0, 0]


def test_manual_mode_does_not_book_a_dividend_twice():
    """A dividend recorded as a $CASH buy plus a $CASH dividend row is one
    payment, not two — the dividend row only classifies the movement."""
    paired = [(BUY, 100.0), (DIVIDEND, 100.0)]
    assert _cash_balance(paired, cash_mode="manual") == 100.0


def test_manual_mode_does_not_book_a_fee_twice():
    paired = [(DEPOSIT, 1000.0), (WITHDRAWAL, 75.0), (FEES, 75.0)]
    assert _cash_balance(paired, cash_mode="manual") == 925.0


def test_auto_mode_still_books_income_on_cash():
    """Auto-mode accounts have no paired movement row, so the income row is the
    only record of the cash arriving."""
    assert _cash_balance([(DIVIDEND, 100.0)], cash_mode="auto") == 100.0
    assert _cash_balance([(INTEREST, 5.0)], cash_mode="auto") == 5.0
    assert _cash_balance([(DEPOSIT, 100.0), (FEES, 10.0)], cash_mode="auto") == 90.0


def test_commission_on_a_cash_row_is_not_charged_twice():
    """A $CASH row mirroring a trade repeats that trade's commission, and the
    charge itself arrives as its own row — only the amount moves the balance."""
    rows = [(DEPOSIT, 1000.0), (SELL, 100.0), (WITHDRAWAL, 2.0)]  # 2.0 = the fee
    assert _cash_balance(rows, cash_mode="manual", commission=1.0) == 898.0
    assert _cash_balance(rows, cash_mode="auto", commission=1.0) == 898.0


def test_plain_cash_movements_are_unaffected_by_mode():
    rows = [(DEPOSIT, 500.0), (BUY, 250.0), (SELL, 100.0), (WITHDRAWAL, 50.0)]
    assert _cash_balance(rows, cash_mode="manual") == 600.0
    assert _cash_balance(rows, cash_mode="auto") == 600.0


# --- "Target quantity" splits (negative Split Ratio) ---


def test_negative_split_ratio_sets_the_position_to_that_quantity():
    """A negative Split Ratio means "the position becomes N shares", for this
    account only — the encoding portfolio_analyzer uses. Skipping it strands the
    difference as a phantom holding the summary has already closed."""
    n = 2
    day = pd.Timestamp("2026-08-10").toordinal()
    qty, _, last_px = _calculate_daily_holdings_chronological_numba(
        date_ordinals_np=np.array([day], dtype=np.int64),
        tx_dates_ordinal_np=np.full(n, day, dtype=np.int64),
        tx_symbols_np=np.array([1, 1], dtype=np.int64),
        tx_accounts_np=np.zeros(n, dtype=np.int64),
        tx_types_np=np.array([BUY, SPLIT], dtype=np.int64),
        tx_quantities_np=np.array([491.0, 0.0], dtype=np.float64),
        tx_prices_np=np.array([30.78, 0.0], dtype=np.float64),
        tx_commissions_np=np.zeros(n, dtype=np.float64),
        tx_split_ratios_np=np.array([0.0, -446.0], dtype=np.float64),
        tx_to_accounts_np=np.full(n, -1, dtype=np.int64),
        tx_totals_np=np.array([491 * 30.78, 0.0], dtype=np.float64),
        num_symbols=2,
        num_accounts=1,
        split_type_id=SPLIT,
        stock_split_type_id=STOCK_SPLIT,
        buy_type_id=BUY,
        deposit_type_id=DEPOSIT,
        sell_type_id=SELL,
        withdrawal_type_id=WITHDRAWAL,
        short_sell_type_id=SHORT_SELL,
        buy_to_cover_type_id=BUY_TO_COVER,
        transfer_type_id=TRANSFER,
        spin_off_type_id=SPIN_OFF,
        fees_type_id=FEES,
        dividend_type_id=DIVIDEND,
        interest_type_id=INTEREST,
        tax_type_id=TAX,
        cash_symbol_id=CASH_SYMBOL_ID,
        stock_qty_close_tolerance=1e-9,
        shortable_symbol_ids=np.zeros(0, dtype=np.int64),
        acc_cash_modes=np.zeros(1, dtype=np.int64),
    )
    assert qty[0, 1, 0] == pytest.approx(446.0)
    # Value is preserved across the re-mark: 491 * 30.78 == 446 * new price.
    assert last_px[0, 1, 0] * 446 == pytest.approx(491 * 30.78)


def test_positive_split_ratio_still_multiplies():
    n = 2
    day = pd.Timestamp("2026-08-10").toordinal()
    qty, _, _ = _calculate_daily_holdings_chronological_numba(
        date_ordinals_np=np.array([day], dtype=np.int64),
        tx_dates_ordinal_np=np.full(n, day, dtype=np.int64),
        tx_symbols_np=np.array([1, 1], dtype=np.int64),
        tx_accounts_np=np.zeros(n, dtype=np.int64),
        tx_types_np=np.array([BUY, SPLIT], dtype=np.int64),
        tx_quantities_np=np.array([100.0, 0.0], dtype=np.float64),
        tx_prices_np=np.array([50.0, 0.0], dtype=np.float64),
        tx_commissions_np=np.zeros(n, dtype=np.float64),
        tx_split_ratios_np=np.array([0.0, 4.0], dtype=np.float64),
        tx_to_accounts_np=np.full(n, -1, dtype=np.int64),
        tx_totals_np=np.array([5000.0, 0.0], dtype=np.float64),
        num_symbols=2,
        num_accounts=1,
        split_type_id=SPLIT,
        stock_split_type_id=STOCK_SPLIT,
        buy_type_id=BUY,
        deposit_type_id=DEPOSIT,
        sell_type_id=SELL,
        withdrawal_type_id=WITHDRAWAL,
        short_sell_type_id=SHORT_SELL,
        buy_to_cover_type_id=BUY_TO_COVER,
        transfer_type_id=TRANSFER,
        spin_off_type_id=SPIN_OFF,
        fees_type_id=FEES,
        dividend_type_id=DIVIDEND,
        interest_type_id=INTEREST,
        tax_type_id=TAX,
        cash_symbol_id=CASH_SYMBOL_ID,
        stock_qty_close_tolerance=1e-9,
        shortable_symbol_ids=np.zeros(0, dtype=np.int64),
        acc_cash_modes=np.zeros(1, dtype=np.int64),
    )
    assert qty[0, 1, 0] == pytest.approx(400.0)


# --- Which column carries the amount on a $CASH row ---


def test_money_in_reads_quantity_money_out_reads_total():
    """When Quantity and Total Amount disagree (a fee baked into one of them),
    the summary reads inflows from Quantity and outflows from Total Amount."""
    # qty=100, total=110 on each row.
    assert _cash_balance([(DEPOSIT, 100.0)], "manual", totals=[110.0]) == 100.0
    assert _cash_balance([(BUY, 100.0)], "manual", totals=[110.0]) == 100.0
    assert _cash_balance([(WITHDRAWAL, 100.0)], "manual", totals=[110.0]) == -110.0
    assert _cash_balance([(SELL, 100.0)], "manual", totals=[110.0]) == -110.0


def test_each_direction_falls_back_to_the_other_column():
    assert _cash_balance([(DEPOSIT, 0.0)], "manual", totals=[110.0]) == 110.0
    assert _cash_balance([(WITHDRAWAL, 100.0)], "manual", totals=[0.0]) == -100.0


def test_dividend_amount_prefers_the_recorded_total():
    """Total Amount is what the broker paid. Shares x per-share rate only
    reconstructs it, and a reinvestment's rounded share count makes the product
    drift (0.13 * 113.4363 = 14.75 for a 14.63 dividend), so it is the fallback."""
    assert _cash_balance(
        [(DIVIDEND, 0.13)], "auto", totals=[14.63], prices=[113.4363]
    ) == pytest.approx(14.63)
    # No total recorded -> reconstruct from shares x rate.
    assert _cash_balance(
        [(DIVIDEND, 0.13)], "auto", totals=[0.0], prices=[113.4363]
    ) == pytest.approx(0.13 * 113.4363)
    # Legacy form: Price/Share holds the amount.
    assert _cash_balance(
        [(DIVIDEND, 0.0)], "auto", totals=[0.0], prices=[20.0]
    ) == pytest.approx(20.0)


# --- Short sales in Auto cash mode ---


def _short_trip_cash(shortable):
    """Cash after a short sale and its cover, with the symbol either shortable
    or not. Symbol id 1; the account is in Auto cash mode."""
    n = 2
    day = pd.Timestamp("2026-08-10").toordinal()
    _, cash, _ = _calculate_daily_holdings_chronological_numba(
        date_ordinals_np=np.array([day], dtype=np.int64),
        tx_dates_ordinal_np=np.full(n, day, dtype=np.int64),
        tx_symbols_np=np.array([1, 1], dtype=np.int64),
        tx_accounts_np=np.zeros(n, dtype=np.int64),
        tx_types_np=np.array([SHORT_SELL, BUY_TO_COVER], dtype=np.int64),
        tx_quantities_np=np.array([20.0, 20.0], dtype=np.float64),
        tx_prices_np=np.array([93.695, 94.94], dtype=np.float64),
        tx_commissions_np=np.zeros(n, dtype=np.float64),
        tx_split_ratios_np=np.zeros(n, dtype=np.float64),
        tx_to_accounts_np=np.full(n, -1, dtype=np.int64),
        tx_totals_np=np.array([1873.90, 1898.80], dtype=np.float64),
        num_symbols=2,
        num_accounts=1,
        split_type_id=SPLIT,
        stock_split_type_id=STOCK_SPLIT,
        buy_type_id=BUY,
        deposit_type_id=DEPOSIT,
        sell_type_id=SELL,
        withdrawal_type_id=WITHDRAWAL,
        short_sell_type_id=SHORT_SELL,
        buy_to_cover_type_id=BUY_TO_COVER,
        transfer_type_id=TRANSFER,
        spin_off_type_id=SPIN_OFF,
        fees_type_id=FEES,
        dividend_type_id=DIVIDEND,
        interest_type_id=INTEREST,
        tax_type_id=TAX,
        cash_symbol_id=CASH_SYMBOL_ID,
        stock_qty_close_tolerance=1e-9,
        shortable_symbol_ids=(
            np.array([1], dtype=np.int64) if shortable else np.zeros(0, dtype=np.int64)
        ),
        acc_cash_modes=np.ones(1, dtype=np.int64),
    )
    return cash[0, 0]


def test_shortable_symbol_does_not_synthesize_cash_for_a_short_trip():
    """A short on a shortable symbol books its cash through its own $CASH rows.
    The summary's shortable branch returns before synthesizing any, so
    synthesizing here would count the proceeds twice and strand a residue in an
    account that has been closed for years."""
    assert _short_trip_cash(shortable=True) == 0.0


def test_non_shortable_symbol_still_synthesizes_cash():
    assert _short_trip_cash(shortable=False) == pytest.approx(20 * 93.695 - 20 * 94.94)
