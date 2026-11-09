"""The chronological holdings kernel and its small helpers.

Split from portfolio_logic.py. The replay runs in Rust (rust/investa_core,
imported as `investa_core`); this module hands it contiguous NumPy inputs. Its
name still says "numba" because callers import it by that name.

The single-date valuation (`_calculate_portfolio_value_at_date_unadjusted` and
its Python and "numba" variants, with the single-date holdings kernel) was
removed on 9 Nov 2026: the history always replays chronologically.
"""

import investa_core
import numpy as np


def _i64(a):
    """An int64, C-contiguous view (or copy) of `a`, as the Rust kernels take."""
    return np.ascontiguousarray(a, dtype=np.int64)


def _f64(a):
    """A float64, C-contiguous view (or copy) of `a`, as the Rust kernels take."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _dividend_amount(qty, price, total_amount):
    """The cash a dividend row pays, read the way the summary reads it.

    Total Amount is the payment whenever it is filled — it is what the broker
    actually paid. Shares x per-share rate only reconstructs it, and on a
    reinvestment the rounded share count makes the product drift, so it is the
    fallback; a bare Price/Share is the legacy form. Mirrors the dividend
    normalization in portfolio_analyzer so the graph and the holdings table
    book the same cash. Computed in Rust (investa_core) alongside the kernels.
    """
    return investa_core.dividend_amount(
        float(qty), float(price), float(total_amount)
    )


def _normalize_series(series):
    """Normalizes a pandas Series to uppercase and trimmed strings."""
    if series.empty:
        return series
    return series.astype(str).str.upper().str.strip()


def _calculate_daily_holdings_chronological_numba(
    date_ordinals_np,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_to_accounts_np,
    tx_accounts_np,
    tx_types_np,
    tx_quantities_np,
    tx_commissions_np,
    tx_split_ratios_np,
    tx_prices_np,
    tx_totals_np,
    num_symbols,
    num_accounts,
    split_type_id,
    stock_split_type_id,
    buy_type_id,
    deposit_type_id,
    sell_type_id,
    withdrawal_type_id,
    short_sell_type_id,
    buy_to_cover_type_id,
    transfer_type_id,
    spin_off_type_id,
    dividend_type_id,
    interest_type_id,
    fees_type_id,
    tax_type_id,
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,
):
    """Holdings, cash and last trade prices at the end of every day in a range.

    Runs in Rust: investa_core.calculate_daily_holdings_chronological, a
    line-for-line port of the Numba kernel that used to live here (frozen in
    tests/reference/). Transactions must be sorted by date. Returns
    (daily_qty[D, S, A], daily_cash[D, A], daily_last_prices[D, S, A]). Note the
    argument order: destination accounts before source accounts.
    """
    return investa_core.calculate_daily_holdings_chronological(
        _i64(date_ordinals_np),
        _i64(tx_dates_ordinal_np),
        _i64(tx_symbols_np),
        _i64(tx_to_accounts_np),
        _i64(tx_accounts_np),
        _i64(tx_types_np),
        _f64(tx_quantities_np),
        _f64(tx_commissions_np),
        _f64(tx_split_ratios_np),
        _f64(tx_prices_np),
        _f64(tx_totals_np),
        int(num_symbols),
        int(num_accounts),
        int(split_type_id),
        int(stock_split_type_id),
        int(buy_type_id),
        int(deposit_type_id),
        int(sell_type_id),
        int(withdrawal_type_id),
        int(short_sell_type_id),
        int(buy_to_cover_type_id),
        int(transfer_type_id),
        int(spin_off_type_id),
        int(dividend_type_id),
        int(interest_type_id),
        int(fees_type_id),
        int(tax_type_id),
        int(cash_symbol_id),
        float(stock_qty_close_tolerance),
        _i64(shortable_symbol_ids),
        _i64(acc_cash_modes),
    )
