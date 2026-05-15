"""
tests/test_cash_mode_equivalence.py

Confirms that Auto-cash mode and Manual-cash mode produce identical results
for holdings, cash balance, realized gains, and external cash flows (which
drive TWR), given the same underlying economic activity.

Also confirms that a common manual-cash data-entry mistake — recording a
paired deposit for every stock buy — suppresses TWR by double-counting
those deposits as external inflows.

Portfolio scenario
------------------
Date        Event
2022-01-05  External deposit  $10,000
2022-01-10  Buy  100 TESTX @ $50  ($5,000)
2022-03-15  Sell  50 TESTX @ $70  ($3,500)
2022-06-01  Dividend from TESTX   $200
2022-08-01  External withdrawal   $2,000

Expected state after all events
--------------------------------
Cash balance : $10,000 - $5,000 + $3,500 + $200 - $2,000 = $6,700
TESTX held   : 100 - 50 = 50 shares  (cost basis $2,500)
Realized gain: 50 × ($70 - $50) = $1,000

External flows for TWR (correct)
---------------------------------
2022-01-05 : +$10,000  (deposit)
2022-08-01 : -$2,000   (withdrawal)
All other dates: $0
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from datetime import date

# ── path setup ────────────────────────────────────────────────────────────────
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from portfolio_logic import (
    CASH_SYMBOL_CSV,
    _calculate_daily_net_cash_flow,
    _calculate_portfolio_value_at_date_unadjusted_numba,
)
from portfolio_analyzer import calculate_fifo_lots_and_gains

# ── constants ─────────────────────────────────────────────────────────────────
ACCOUNT = "TestBrokerage"
USD = "USD"
TESTX_PRICE = 70.0   # constant price used for valuation assertions
TARGET_DATE = date(2022, 12, 31)

_COLS = [
    "Date", "Type", "Symbol", "Quantity", "Price/Share",
    "Total Amount", "Commission", "Account", "Split Ratio",
    "Note", "Local Currency", "To Account", "Tags", "ExternalID", "user_id",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _tx(date_str, typ, symbol, qty, price, total=None, commission=0.0):
    if total is None:
        total = abs(qty * price)
    return {
        "Date": pd.Timestamp(date_str),
        "Type": typ,
        "Symbol": symbol,
        "Quantity": float(qty),
        "Price/Share": float(price),
        "Total Amount": float(total),
        "Commission": float(commission),
        "Account": ACCOUNT,
        "Split Ratio": np.nan,
        "Note": "",
        "Local Currency": USD,
        "To Account": np.nan,
        "Tags": np.nan,
        "ExternalID": np.nan,
        "user_id": 1,
    }


def _build_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLS)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.reset_index(drop=True)
    df["original_index"] = df.index
    return df


def _mappings(df: pd.DataFrame):
    """Build string→id maps required by Numba valuation functions."""
    all_syms = sorted(df["Symbol"].unique().tolist())
    if CASH_SYMBOL_CSV not in all_syms:
        all_syms.append(CASH_SYMBOL_CSV)
    symbol_to_id = {s: i for i, s in enumerate(all_syms)}
    id_to_symbol = {i: s for s, i in symbol_to_id.items()}

    all_accs = sorted(df["Account"].unique().tolist())
    account_to_id = {a: i for i, a in enumerate(all_accs)}
    id_to_account = {i: a for a, i in account_to_id.items()}

    all_types = sorted(df["Type"].str.lower().str.strip().unique().tolist())
    type_to_id = {t: i for i, t in enumerate(all_types)}

    all_curs = sorted(df["Local Currency"].unique().tolist())
    currency_to_id = {c: i for i, c in enumerate(all_curs)}
    id_to_currency = {i: c for c, i in currency_to_id.items()}

    return (
        symbol_to_id, id_to_symbol,
        account_to_id, id_to_account,
        type_to_id,
        currency_to_id, id_to_currency,
    )


# ── transaction sets ──────────────────────────────────────────────────────────

def _auto_cash_df() -> pd.DataFrame:
    """
    Auto-cash: only record the stock side of each trade.
    Cash is computed implicitly by the engine from buy/sell totals.

    Dividend convention (auto-cash): qty=0, Price/Share = total dividend amount.
    The engine uses abs(Price/Share) as the cash delta for dividends.
    """
    return _build_df([
        _tx("2022-01-05", "deposit",    CASH_SYMBOL_CSV, 10000, 1.0),
        _tx("2022-01-10", "buy",        "TESTX",           100, 50.0, total=5000),
        _tx("2022-03-15", "sell",       "TESTX",            50, 70.0, total=3500),
        _tx("2022-06-01", "dividend",   "TESTX",             0, 200.0, total=200),
        _tx("2022-08-01", "withdrawal", CASH_SYMBOL_CSV,  2000, 1.0),
    ])


def _manual_cash_correct_df() -> pd.DataFrame:
    """
    Manual-cash (correct): every stock buy/sell/dividend has an explicit
    $CASH counterpart using type='buy'/'sell' (NOT 'deposit'/'withdrawal').

    This means no paired deposits appear as external flows.
    TWR should be identical to auto-cash.
    """
    return _build_df([
        # Real external deposit
        _tx("2022-01-05", "deposit",  CASH_SYMBOL_CSV, 10000, 1.0),
        # Buy TESTX: pay with "sell $CASH" (type=sell, NOT deposit)
        _tx("2022-01-10", "buy",  "TESTX",           100, 50.0, total=5000),
        _tx("2022-01-10", "sell", CASH_SYMBOL_CSV,  5000, 1.0),
        # Sell TESTX: receive via "buy $CASH" (type=buy, NOT deposit)
        _tx("2022-03-15", "sell", "TESTX",           50, 70.0, total=3500),
        _tx("2022-03-15", "buy",  CASH_SYMBOL_CSV, 3500, 1.0),
        # Dividend: receive via "buy $CASH" (type=buy, NOT deposit)
        _tx("2022-06-01", "dividend", "TESTX",      200, 1.0),
        _tx("2022-06-01", "buy",  CASH_SYMBOL_CSV,  200, 1.0),
        # Real external withdrawal
        _tx("2022-08-01", "withdrawal", CASH_SYMBOL_CSV, 2000, 1.0),
    ])


def _manual_cash_paired_deposits_df() -> pd.DataFrame:
    """
    Manual-cash (incorrect): user records a 'deposit' for every stock buy.
    Those paired deposits are type='deposit', so the TWR engine counts them
    as external inflows, inflating the denominator and suppressing TWR.
    """
    return _build_df([
        # Real external deposit
        _tx("2022-01-05", "deposit",  CASH_SYMBOL_CSV, 10000, 1.0),
        # BUG: extra 'deposit' paired with buy (should be 'sell $CASH' instead)
        _tx("2022-01-10", "deposit", CASH_SYMBOL_CSV,  5000, 1.0),  # ← incorrect
        _tx("2022-01-10", "buy",     "TESTX",           100, 50.0, total=5000),
        _tx("2022-01-10", "sell",    CASH_SYMBOL_CSV,  5000, 1.0),
        # Sell and dividend: correct
        _tx("2022-03-15", "sell", "TESTX",           50, 70.0, total=3500),
        _tx("2022-03-15", "buy",  CASH_SYMBOL_CSV, 3500, 1.0),
        _tx("2022-06-01", "dividend", "TESTX",      200, 1.0),
        _tx("2022-06-01", "buy",  CASH_SYMBOL_CSV,  200, 1.0),
        # Real external withdrawal
        _tx("2022-08-01", "withdrawal", CASH_SYMBOL_CSV, 2000, 1.0),
    ])


# ── external flow helper ──────────────────────────────────────────────────────

def _net_flow(df: pd.DataFrame, target: date) -> float:
    flow, _ = _calculate_daily_net_cash_flow(
        target_date=target,
        transactions_df=df,
        target_currency=USD,
        historical_fx_yf={},
        account_currency_map={ACCOUNT: USD},
        default_currency=USD,
        processed_warnings=set(),
        included_accounts=None,
        historical_prices_yf_unadjusted=None,
        internal_to_yf_map={},
    )
    return 0.0 if pd.isna(flow) else float(flow)


# ── portfolio value helper ────────────────────────────────────────────────────

def _portfolio_value(df: pd.DataFrame, cash_mode: str) -> float:
    """Run the Numba valuation at TARGET_DATE using a fixed TESTX price."""
    maps = _mappings(df)
    (symbol_to_id, id_to_symbol, account_to_id, id_to_account,
     type_to_id, currency_to_id, id_to_currency) = maps

    unadjusted = {
        "TESTX": pd.DataFrame(
            {"price": [TESTX_PRICE]},
            index=pd.to_datetime([TARGET_DATE], utc=True),
        )
    }

    value, _ = _calculate_portfolio_value_at_date_unadjusted_numba(
        target_date=TARGET_DATE,
        transactions_df=df,
        historical_prices_yf_unadjusted=unadjusted,
        historical_fx_yf={},
        target_currency=USD,
        internal_to_yf_map={"TESTX": "TESTX"},
        account_currency_map={ACCOUNT: USD},
        default_currency=USD,
        manual_overrides_dict=None,
        processed_warnings=set(),
        symbol_to_id=symbol_to_id,
        id_to_symbol=id_to_symbol,
        account_to_id=account_to_id,
        id_to_account=id_to_account,
        type_to_id=type_to_id,
        currency_to_id=currency_to_id,
        id_to_currency=id_to_currency,
        included_accounts=None,
        account_cash_mode_map={ACCOUNT: cash_mode},
    )
    return float(value)


# ── realized gains helper ─────────────────────────────────────────────────────

def _total_realized_gain(df: pd.DataFrame) -> float:
    """FIFO realized gain in USD."""
    gains_df, _ = calculate_fifo_lots_and_gains(
        transactions_df=df,
        display_currency=USD,
        historical_fx_yf={},
        default_currency=USD,
        shortable_symbols=set(),
    )
    if gains_df.empty:
        return 0.0
    col = "Realized Gain (Display)"
    return float(gains_df[col].sum()) if col in gains_df.columns else 0.0


# ── Tests: external cash flows ────────────────────────────────────────────────

KEY_DATES = [
    date(2022, 1, 5),
    date(2022, 1, 10),
    date(2022, 3, 15),
    date(2022, 6, 1),
    date(2022, 8, 1),
]


class TestExternalFlows:
    """
    External cash flows are determined solely by Type='deposit'/'withdrawal'
    transactions. They drive the TWR denominator.
    """

    def test_deposit_day_same_in_both_modes(self):
        assert _net_flow(_auto_cash_df(), date(2022, 1, 5)) == pytest.approx(10_000, rel=1e-6)
        assert _net_flow(_manual_cash_correct_df(), date(2022, 1, 5)) == pytest.approx(10_000, rel=1e-6)

    def test_buy_day_zero_flow_in_correct_manual_cash(self):
        """'sell $CASH' (type=sell) does NOT register as an external flow."""
        assert _net_flow(_auto_cash_df(), date(2022, 1, 10)) == pytest.approx(0.0, abs=1e-6)
        assert _net_flow(_manual_cash_correct_df(), date(2022, 1, 10)) == pytest.approx(0.0, abs=1e-6)

    def test_sell_day_zero_flow(self):
        """'buy $CASH' (type=buy) does NOT register as an external flow."""
        assert _net_flow(_auto_cash_df(), date(2022, 3, 15)) == pytest.approx(0.0, abs=1e-6)
        assert _net_flow(_manual_cash_correct_df(), date(2022, 3, 15)) == pytest.approx(0.0, abs=1e-6)

    def test_dividend_day_zero_flow(self):
        """Dividend 'buy $CASH' does NOT register as an external flow."""
        assert _net_flow(_auto_cash_df(), date(2022, 6, 1)) == pytest.approx(0.0, abs=1e-6)
        assert _net_flow(_manual_cash_correct_df(), date(2022, 6, 1)) == pytest.approx(0.0, abs=1e-6)

    def test_withdrawal_day_same_in_both_modes(self):
        assert _net_flow(_auto_cash_df(), date(2022, 8, 1)) == pytest.approx(-2_000, rel=1e-6)
        assert _net_flow(_manual_cash_correct_df(), date(2022, 8, 1)) == pytest.approx(-2_000, rel=1e-6)

    def test_all_dates_match_auto_vs_correct_manual(self):
        for d in KEY_DATES:
            auto_flow   = _net_flow(_auto_cash_df(), d)
            manual_flow = _net_flow(_manual_cash_correct_df(), d)
            assert auto_flow == pytest.approx(manual_flow, abs=1e-4), (
                f"Flow mismatch on {d}: auto={auto_flow}, manual={manual_flow}"
            )

    def test_paired_deposit_creates_spurious_flow_on_buy_day(self):
        """
        Incorrect manual-cash: 'deposit $CASH $5,000' on the buy day shows
        as a $5,000 external inflow — which correct recording avoids.
        """
        bad_flow = _net_flow(_manual_cash_paired_deposits_df(), date(2022, 1, 10))
        ok_flow  = _net_flow(_manual_cash_correct_df(), date(2022, 1, 10))
        assert bad_flow == pytest.approx(5_000, rel=1e-6)
        assert ok_flow  == pytest.approx(0.0, abs=1e-6)

    def test_total_inflows_differ_by_paired_deposit_amount(self):
        inflows = lambda df: sum(max(0, _net_flow(df, d)) for d in KEY_DATES)
        auto_in  = inflows(_auto_cash_df())
        ok_in    = inflows(_manual_cash_correct_df())
        bad_in   = inflows(_manual_cash_paired_deposits_df())
        assert auto_in == pytest.approx(ok_in, rel=1e-6)           # same ✓
        assert bad_in  == pytest.approx(ok_in + 5_000, rel=1e-6)   # $5k extra ✗


# ── Tests: portfolio value (cash + stocks) ────────────────────────────────────

class TestPortfolioValue:
    """
    Total value = 50 TESTX × $70 + $6,700 cash = $10,200.
    Both correct modes must produce this.

    The paired-deposit (incorrect) version inflates value by $5,000:
    the 'sell $CASH' already accounts for the TESTX purchase cost, so
    the extra 'deposit $CASH' leaves $5,000 of phantom cash in the account.
    """

    EXPECTED = 50 * TESTX_PRICE + 6_700.0   # $10,200

    def test_auto_cash_value(self):
        assert _portfolio_value(_auto_cash_df(), "Auto") == pytest.approx(self.EXPECTED, rel=1e-4)

    def test_correct_manual_cash_value(self):
        assert _portfolio_value(_manual_cash_correct_df(), "Manual") == pytest.approx(self.EXPECTED, rel=1e-4)

    def test_auto_equals_correct_manual(self):
        val_auto   = _portfolio_value(_auto_cash_df(), "Auto")
        val_manual = _portfolio_value(_manual_cash_correct_df(), "Manual")
        assert val_auto == pytest.approx(val_manual, rel=1e-4)

    def test_paired_deposit_inflates_value(self):
        """
        The extra 'deposit $CASH $5,000' on the buy day is NOT cancelled by the
        'sell $CASH $5,000' — that sell already represents the TESTX purchase cost.
        The spurious deposit therefore leaves $5,000 of unearned cash in the
        account, inflating portfolio value by exactly $5,000.

        Correct value: $10,200  (auto-cash or correct manual-cash)
        Bad value:     $15,200  (paired-deposit manual-cash)
        """
        val_auto = _portfolio_value(_auto_cash_df(), "Auto")
        val_bad  = _portfolio_value(_manual_cash_paired_deposits_df(), "Manual")
        assert val_bad == pytest.approx(val_auto + 5_000, rel=1e-4)


# ── Tests: realized gains ─────────────────────────────────────────────────────

class TestRealizedGains:
    """
    FIFO cost-basis matching uses only stock buy/sell transactions.
    $CASH buy/sell artifacts produce zero gain. Both modes must yield $1,000.
    """

    EXPECTED_GAIN = 1_000.0

    def test_auto_cash_realized_gain(self):
        assert _total_realized_gain(_auto_cash_df()) == pytest.approx(self.EXPECTED_GAIN, rel=1e-4)

    def test_correct_manual_cash_realized_gain(self):
        assert _total_realized_gain(_manual_cash_correct_df()) == pytest.approx(self.EXPECTED_GAIN, rel=1e-4)

    def test_paired_deposit_does_not_change_realized_gain(self):
        gain_auto = _total_realized_gain(_auto_cash_df())
        gain_bad  = _total_realized_gain(_manual_cash_paired_deposits_df())
        assert gain_auto == pytest.approx(gain_bad, rel=1e-4)


# ── Tests: TWR formula impact (analytical) ────────────────────────────────────

class TestTWRImpact:
    """
    Shows analytically that a paired deposit on the buy day suppresses that
    day's TWR sub-period return, and that this effect compounds over N buys.
    """

    def test_paired_deposit_makes_buy_day_return_negative(self):
        """
        Scenario: portfolio value is $10,000 before and after the buy
        (stock bought at market price = cost).

        Correct:   net_flow = 0   → return = 0 / 10,000 = 0.0 %
        Incorrect: net_flow = +5,000 (paired deposit)
                   gain = 10,000 - 10,000 - 5,000 = -5,000
                   denom = 10,000 + 5,000 = 15,000
                   return = -5,000 / 15,000 ≈ -33.3 %
        """
        prev_value = 10_000.0
        curr_value = 10_000.0

        def sub_period(net_flow):
            gain  = curr_value - prev_value - net_flow
            denom = prev_value + max(0.0, net_flow)
            return gain / denom if denom > 0.1 else 0.0

        r_correct = sub_period(0.0)
        r_bad     = sub_period(5_000.0)

        assert r_correct == pytest.approx(0.0, abs=1e-9)
        assert r_bad     == pytest.approx(-5_000 / 15_000, rel=1e-6)
        assert r_bad < r_correct

    def test_twr_suppression_compounds_with_multiple_buys(self):
        """
        Five buys of $5,000 each, portfolio value unchanged on each buy day.
        Correct TWR factor: 1.0 (0% return × 5 days)
        Incorrect: each day multiplied by (1 - 5000/15000) = 10/15
        After 5 days: (10/15)^5 ≈ 0.132 → TWR ≈ −86.8%
        """
        n_buys   = 5
        buy_size = 5_000.0
        V        = 10_000.0

        factor_correct = 1.0
        factor_bad     = (1.0 - buy_size / (V + buy_size)) ** n_buys

        twr_correct = factor_correct - 1   # 0%
        twr_bad     = factor_bad - 1

        assert twr_correct == pytest.approx(0.0, abs=1e-9)
        assert twr_bad < twr_correct
        expected = (V / (V + buy_size)) ** n_buys - 1
        assert twr_bad == pytest.approx(expected, rel=1e-9)
