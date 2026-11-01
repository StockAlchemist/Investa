# Frozen reference: _value_daily_holdings_vectorized from src/portfolio_history.py
# at commit 6386bd79, before the Rust port. tests/test_rust_history_parity.py
# holds rust/investa_core/src/history.rs (and the shim around it) to it,
# bitwise. Never edit this to follow a change in Rust.
# ruff: noqa
import logging
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import CASH_SYMBOL_CSV
from portfolio_history import (
    _fund_nav_series,
    align_prices_to_grid,
    manual_prices_by_symbol_id,
)


def _value_daily_holdings_vectorized(
    date_range: pd.DatetimeIndex,
    daily_holdings_qty_np: np.ndarray,
    daily_last_prices_np: Optional[np.ndarray],
    daily_cash_balances_np: np.ndarray,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    account_currency_map: Dict[str, str],
    id_to_symbol: Dict[int, str],
    id_to_account: Dict[int, str],
    internal_to_yf_map: Dict[str, str],
    account_ids_to_include_set: Set[int],
    default_currency: str,
    interval: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.Series, bool, str]:
    """
    Vectorized valuation of holdings and cash.
    Returns (daily_value_series, has_errors, status_msg).
    """
    num_days, num_symbols, num_accounts = daily_holdings_qty_np.shape
    has_errors = False
    status_msg = ""

    # 1. Prepare aligned PRICES array (N_days, N_symbols)
    # Initialize with NaNs
    daily_prices_aligned = np.full((num_days, num_symbols), np.nan, dtype=np.float64)

    # Collect price series by sym_id for bulk reindexing (avoids per-symbol
    # .reindex() calls that each iterate the full DatetimeIndex — the #2
    # profiling hotspot at ~0.55s).
    price_series_by_id: dict = {}
    for sym_id in range(num_symbols):
        symbol = id_to_symbol.get(sym_id)
        if symbol == CASH_SYMBOL_CSV:
            continue

        yf_symbol = internal_to_yf_map.get(symbol)
        if yf_symbol and yf_symbol in historical_prices_yf_unadjusted:
            price_df = historical_prices_yf_unadjusted[yf_symbol]
            if not price_df.empty:
                col_to_use = None
                if "price" in price_df.columns:
                    col_to_use = "price"
                elif "Close" in price_df.columns:
                    col_to_use = "Close"
                elif "Adj Close" in price_df.columns:
                    col_to_use = "Adj Close"

                if col_to_use:
                    ps = price_df[col_to_use]
                    # Ensure UTC (may already be done by market_data.py)
                    if (
                        not isinstance(ps.index, pd.DatetimeIndex)
                        or ps.index.tz is None
                    ):
                        ps = ps.copy()
                        ps.index = pd.to_datetime(ps.index, utc=True)
                    price_series_by_id[sym_id] = ps

    # 1b. Published NAVs for holdings with no market feed.
    #
    # Thai mutual funds have no ticker and no quote, so they used to fall
    # straight through to the single hand-entered override below — which meant
    # a fund held since 2010 was valued at today's NAV on every day of its
    # history. Flat, and wrong by whatever the fund had actually done since.
    # The SEC publishes the daily series; fund_nav holds it.
    #
    # These join price_series_by_id rather than being written to the matrix
    # directly so they get the same grid alignment as everything else — last
    # observation carried forward across weekends and holidays, never
    # interpolated.
    nav_symbol_ids = {
        sym_id: symbol
        for sym_id, symbol in id_to_symbol.items()
        if symbol and symbol != CASH_SYMBOL_CSV and sym_id not in price_series_by_id
    }
    if nav_symbol_ids:
        for sym_id, series in _fund_nav_series(nav_symbol_ids, date_range).items():
            price_series_by_id[sym_id] = series

    if price_series_by_id:
        price_matrix = align_prices_to_grid(price_series_by_id, date_range)
        for sym_id in price_series_by_id:
            daily_prices_aligned[:, sym_id] = price_matrix[sym_id].values

    # 2. Where neither the market nor a published NAV has anything to say, use
    # the NAV the user entered.
    # This is the same fallback order the summary applies (portfolio_analyzer:
    # market price, then the manual override, then the last transaction price),
    # so a fund with no price feed is worth the same on the graph as it is in
    # the holdings table. Without it these holdings sat frozen at whatever they
    # last traded at, and the chart drifted away from the headline total.
    for sym_id, manual_price in manual_prices_by_symbol_id(
        manual_overrides_dict, id_to_symbol
    ).items():
        column = daily_prices_aligned[:, sym_id]
        np.copyto(column, manual_price, where=np.isnan(column))

    # 3. Prepare aligned FX array (N_days, N_accounts)
    daily_fx_aligned = np.full((num_days, num_accounts), np.nan, dtype=np.float64)

    # id_to_account holds normalized (uppercased/trimmed) names, but the
    # configured map keeps display case (e.g. "Kim Eng") — normalize the keys
    # so mixed-case accounts don't silently fall back to default_currency.
    account_currency_map_normalized = {
        str(k).upper().strip(): v for k, v in account_currency_map.items()
    }

    # Cache FX series to avoid re-fetching for same currency
    currency_fx_series_cache = {}

    # Pre-fetch Target Currency Rate vs USD (Series)
    # Target Rate = Target / USD
    target_rate_series = None
    target_curr_upper = target_currency.upper()

    if target_curr_upper == "USD":
        target_rate_series = pd.Series(1.0, index=date_range)
    else:
        # Fetch Target=X (e.g. THB=X means THB per USD)
        target_pair = f"{target_curr_upper}=X"
        # FALLBACK: Try USDCURR=X if CURR=X missing, or vice versa
        if target_pair not in historical_fx_yf:
            if target_curr_upper == "THB" and "USDTHB=X" in historical_fx_yf:
                target_pair = "USDTHB=X"
            elif (
                target_curr_upper != "THB"
                and f"USD{target_curr_upper}=X" in historical_fx_yf
            ):
                target_pair = f"USD{target_curr_upper}=X"

        if target_pair in historical_fx_yf:
            t_df = historical_fx_yf[target_pair]
            if not t_df.empty:
                col_to_use = None
                if "price" in t_df.columns:
                    col_to_use = "price"
                elif "Close" in t_df.columns:
                    col_to_use = "Close"
                elif "Adj Close" in t_df.columns:
                    col_to_use = "Adj Close"
                elif "rate" in t_df.columns:
                    col_to_use = "rate"

                if col_to_use:
                    # Reindex to date_range with ffill
                    # Reindex to date_range with ffill then bfill to cover start of history
                    t_series = t_df[col_to_use].copy()
                    t_series.index = pd.to_datetime(t_series.index, utc=True)

                    # --- ORIENTATION CHECK ---
                    # Ensure rate is Local / USD.
                    # If it's THB=X or USDTHB=X (~35), it's already Local/USD.
                    # If it was returned as a Major pair (e.g. EURUSD=X ~1.08), it's USD/Local -> Invert.
                    # Note: My analysis shows yfinance EUR=X is already EUR/USD (~0.92).
                    # But we keep this check for robustness if a USD... pair was fetched.
                    if target_pair.endswith("USD=X") and not target_pair.startswith(
                        "USD"
                    ):
                        # e.g. EURUSD=X -> USD per EUR -> Invert to get EUR per USD
                        t_series = 1.0 / t_series

                    target_rate_series = (
                        t_series.reindex(date_range)
                        .interpolate(method="linear")
                        .ffill()
                        .bfill()
                    )

    # If target rate missing, we can't convert anything (unless local==target)
    # Ensure it's not None for math consistency, fill with NaN if missing
    if target_rate_series is None:
        target_rate_series = pd.Series(np.nan, index=date_range)

    for acc_id in range(num_accounts):
        if acc_id not in account_ids_to_include_set:
            continue

        acc_name = id_to_account.get(acc_id)
        local_curr = account_currency_map_normalized.get(acc_name, default_currency)
        local_curr_upper = local_curr.upper()

        if local_curr in currency_fx_series_cache:
            daily_fx_aligned[:, acc_id] = currency_fx_series_cache[local_curr]
            continue

        # Build aligned series
        if local_curr == target_currency:
            rates = np.ones(num_days, dtype=np.float64)
            daily_fx_aligned[:, acc_id] = rates
            currency_fx_series_cache[local_curr] = rates
            continue

        # Calculate Cross Rate: (Target / USD) / (Local / USD)
        # We need Local Rate = Local / USD
        local_rate_series = None
        if local_curr_upper == "USD":
            local_rate_series = pd.Series(1.0, index=date_range)
        else:
            local_pair = f"{local_curr_upper}=X"
            # FALLBACK
            if local_pair not in historical_fx_yf:
                if local_curr_upper == "THB" and "USDTHB=X" in historical_fx_yf:
                    local_pair = "USDTHB=X"
                elif (
                    local_curr_upper != "THB"
                    and f"USD{local_curr_upper}=X" in historical_fx_yf
                ):
                    local_pair = f"USD{local_curr_upper}=X"

            if local_pair in historical_fx_yf:
                l_df = historical_fx_yf[local_pair]
                if not l_df.empty:
                    col_to_use = None
                    if "price" in l_df.columns:
                        col_to_use = "price"
                    elif "Close" in l_df.columns:
                        col_to_use = "Close"
                    elif "Adj Close" in l_df.columns:
                        col_to_use = "Adj Close"
                    elif "rate" in l_df.columns:
                        col_to_use = "rate"

                    if col_to_use:
                        l_series = l_df[col_to_use].copy()
                        # FIX: Ensure UTC awareness and DO NOT Normalize (preserves intraday/hourly timestamps)
                        l_series.index = pd.to_datetime(l_series.index, utc=True)

                        # --- ORIENTATION CHECK ---
                        if local_pair.endswith("USD=X") and not local_pair.startswith(
                            "USD"
                        ):
                            l_series = 1.0 / l_series

                        # FIX: Aggressive backfill for local rates too, but smooth it
                        local_rate_series = (
                            l_series.reindex(date_range)
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )

        if local_rate_series is None:
            # Missing local rate data - Default to 1.0 to prevent NaN propagation
            rates = np.ones(num_days, dtype=np.float64)
            # logging.warning(f"Hist Val: Missing FX rates for {local_curr}, defaulting to 1.0")
            has_errors = True
        else:
            # Compute cross rate
            # Handle division by zero or NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                # aligned_target / aligned_local
                cross_rates = target_rate_series / local_rate_series

                # Final safety: If any cross rate is STILL NaN, it means one of the series was all NaN
                # or had persistent gaps. Try to bfill again to be sure.
                if pd.Series(cross_rates).isna().any():
                    # pandas 3 compatibility (a no-op under pandas 2): a
                    # writable copy, as the NaN fill below writes into it.
                    cross_rates = pd.Series(cross_rates).bfill().ffill().to_numpy(copy=True)

            rates = cross_rates
            # Check for NaNs and ONLY fill with 1.0 as a catastrophic last resort
            # if both target and local rates were completely missing.
            mask_nan_fx = np.isnan(rates)
            if mask_nan_fx.any():
                rates[mask_nan_fx] = 1.0

        daily_fx_aligned[:, acc_id] = rates
        currency_fx_series_cache[local_curr] = rates

    # 4. Calculate Market Value (Stocks)
    # Broadcast Arrays to (Days, Syms, Accs)
    # P_3d = (D, S, 1)
    P_3d = daily_prices_aligned[:, :, np.newaxis]

    # Handle Price Fallback in 3D
    if daily_last_prices_np is not None:
        # P_full = (D, S, A)
        P_full = np.broadcast_to(P_3d, daily_holdings_qty_np.shape).copy()
        mask_nan = np.isnan(P_full)

        # --- NEW: Identify and Log symbols that are missing market data ---
        if mask_nan.any():
            nan_indices = np.where(mask_nan)
            # Just log once per symbol if it has significant gaps
            unique_nan_sym_ids = np.unique(nan_indices[1])
            for s_id in unique_nan_sym_ids:
                sym_name = id_to_symbol.get(s_id, "Unknown")
                if sym_name != CASH_SYMBOL_CSV:
                    logging.debug(
                        f"Valuation: Symbol '{sym_name}' missing market data. Using fallback."
                    )

        P_full[mask_nan] = daily_last_prices_np[mask_nan]

        # --- NEW: Final fallback for symbols with NO market data and NO transaction price ---
        # (Though buy transactions should always have a price)
        # If still NaN, we might have a transfer or split with missing price.
        # Use 1.0 as a catastrophic fallback to avoid total value NaN if possible,
        # but only if quantity > 0. Actually, better to let it be NaN and ffill later.

        V_stocks = daily_holdings_qty_np * P_full * daily_fx_aligned[:, np.newaxis, :]
    else:
        V_stocks = daily_holdings_qty_np * P_3d * daily_fx_aligned[:, np.newaxis, :]

    # --- CRITICAL FIX: Prevent 0.0 * NaN = NaN ---
    # Ensure zero quantity always results in zero value, regardless of Price or FX NaNs
    V_stocks[daily_holdings_qty_np == 0] = 0.0

    # Mask out excluded accounts
    acc_mask = np.zeros(num_accounts, dtype=bool)
    for acc_id in account_ids_to_include_set:
        acc_mask[acc_id] = True

    # Zeros for excluded
    V_stocks[:, :, ~acc_mask] = 0.0

    # Sum over Symbols and Accounts
    total_stock_value = np.sum(V_stocks, axis=(1, 2))

    # 5. Calculate Cash Value
    V_cash = daily_cash_balances_np * daily_fx_aligned
    # --- CRITICAL FIX: Prevent 0.0 * NaN = NaN for Cash ---
    V_cash[daily_cash_balances_np == 0] = 0.0

    V_cash[:, ~acc_mask] = 0.0

    total_cash_value = np.sum(V_cash, axis=1)

    # 6. Total
    total_value = total_stock_value + total_cash_value

    daily_value_series = pd.Series(total_value, index=date_range)

    # --- FIX: Avoid zeroing out portfolio value due to missing data ---
    # We use linear interpolation to spread gains/losses over gaps,
    # which prevents "price restoration" spikes from ruining TWR and Volatility.
    if daily_value_series.isna().any():
        # Identify which dates are missing for logging
        missing_dates = daily_value_series.index[daily_value_series.isna()]
        logging.warning(
            f"Valuation: Missing data for {len(missing_dates)} days. Using linear interpolation to smooth jumps."
        )
        status_msg += " (partial data interpolated)"
        daily_value_series = (
            daily_value_series.interpolate(method="linear").ffill().fillna(0.0)
        )

    return daily_value_series, has_errors, status_msg
