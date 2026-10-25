# Frozen reference: get_historical_price from src/finutils.py at commit fed1eaa9,
# before the Rust port. tests/test_rust_lookup_parity.py holds the PriceTable
# path (rust/investa_core/src/asof.rs) to it, bitwise. Never edit this to
# follow a change in Rust.
# ruff: noqa
import logging
from datetime import date, datetime
from typing import Dict

import numpy as np
import pandas as pd


def get_historical_price(
    symbol_key: str, target_date: date, prices_dict: Dict[str, pd.DataFrame]
) -> float:
    """
    Gets the historical price for a symbol on a specific date, using forward fill for missing dates.

    Looks up the symbol (which can be a YF ticker like 'AAPL' or an FX pair like 'EUR=X')
    in the provided dictionary of price DataFrames. Finds the price for the `target_date`.
    If the exact date is missing, it uses the price from the most recent previous date available
    (forward fill).

    Args:
        symbol_key (str): The key (YF ticker or FX pair string) to look up in `prices_dict`.
        target_date (date): The specific date for which the price is required.
        prices_dict (Dict[str, pd.DataFrame]): A dictionary where keys are symbol keys (str) and
            values are DataFrames indexed by date objects, containing a 'price' column.

    Returns:
        float: The price for the symbol on the target date (or the last available price
               before it). Returns np.nan if the symbol is not found, the date is before
               the first available price point, or an error occurs during lookup.
    """
    if target_date is None or not isinstance(
        target_date, (date, datetime, pd.Timestamp)
    ):
        return np.nan

    if symbol_key not in prices_dict or prices_dict[symbol_key].empty:
        # logging.debug(f"get_historical_price: {symbol_key} missing or empty.")
        return np.nan

    # Avoid modifying the original DataFrame in prices_dict (it might be shared)
    df = prices_dict[symbol_key]

    # DEBUG: Trace THB calls - REMOVED

    try:
        # Optimization: Use asof for efficient lookup
        # If not sorted, sort into a NEW object (do not modify inplace)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Try asof with original target_date
        try:
            # Force target_date to Timestamp for more reliable awareness checks in pandas
            target_ts = pd.Timestamp(target_date)

            # Align awareness and type BEFORE calling asof
            if isinstance(df.index, pd.DatetimeIndex):
                idx_tz = df.index.tz
                target_tz = target_ts.tz

                if idx_tz is not None and target_tz is None:
                    target_ts = target_ts.tz_localize(idx_tz)
                elif idx_tz is None and target_tz is not None:
                    target_ts = target_ts.tz_localize(None)
                elif idx_tz is not None and target_tz is not None:
                    target_ts = target_ts.tz_convert(idx_tz)
            elif (
                len(df.index) > 0
                and isinstance(df.index[0], date)
                and not isinstance(df.index[0], datetime)
            ):
                # index is likely date objects, asof(Timestamp) might fail.
                # Convert target_ts back to date for lookup if index is date objects.
                target_ts = target_ts.date()

            res = df.asof(target_ts)

        except (TypeError, ValueError) as e_asof:
            logging.debug(
                f"AsOf initial fail for {symbol_key} on {target_date}: {e_asof}"
            )
            # Fallback: Try converting to Timestamp
            ts_target = pd.Timestamp(target_date)

            # Handle Timezone if index is tz-aware
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                if ts_target.tz is None:
                    ts_target = ts_target.tz_localize(df.index.tz)

            res = df.asof(ts_target)

        # Extract price from result
        price = np.nan
        if isinstance(res, pd.Series):
            price = res.get("price")
            if price is None or pd.isna(price):
                price = res.get("Adj Close")
            if price is None or pd.isna(price):
                price = res.get("Close")
        elif isinstance(res, pd.DataFrame):
            # Duplicate index case
            if not res.empty:
                if "price" in res.columns:
                    price = res["price"].iloc[-1]
                elif "Adj Close" in res.columns:
                    price = res["Adj Close"].iloc[-1]
                elif "Close" in res.columns:
                    price = res["Close"].iloc[-1]
        elif pd.notna(res):
            pass  # Scalar?

        if price is None or pd.isna(price):
            # Check if target_date is before the start of the data (Backfill Strategy)
            # Use Timestamps for robust comparison
            try:
                first_ts = pd.Timestamp(df.index[0])
                target_ts_eval = pd.Timestamp(target_date)

                # Align awareness for comparison
                if first_ts.tz is not None and target_ts_eval.tz is None:
                    target_ts_eval = target_ts_eval.tz_localize(first_ts.tz)
                elif first_ts.tz is None and target_ts_eval.tz is not None:
                    target_ts_eval = target_ts_eval.tz_localize(None)

                if target_ts_eval < first_ts:
                    col = "price"
                    if "price" not in df.columns:
                        if "Adj Close" in df.columns:
                            col = "Adj Close"
                        elif "Close" in df.columns:
                            col = "Close"

                    backfill_price = df[col].iloc[0]
                    if pd.notna(backfill_price):
                        # logging.debug(f"BACKFILL: {symbol_key} using {backfill_price} for {target_date}")
                        return float(backfill_price)
            except Exception as e_comp:
                logging.debug(f"Comparison fail in get_historical_price: {e_comp}")

        return float(price) if pd.notna(price) else np.nan

    except Exception as e:
        logging.error(
            f"ERROR getting historical price for {symbol_key} on {target_date}: {e}"
        )
        return np.nan


