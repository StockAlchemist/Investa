"""The daily price reads before the Rust reader (phase 8), frozen as plain
functions: `self` is a market_db.MarketDatabase. Used only by
tests/test_rust_ohlcv_parity.py.
"""

from datetime import date
from typing import Dict, List

import pandas as pd

from market_db import ADJUST_SPLIT, BASIS_SPLIT_ADJ, VALID_ADJUSTMENTS


def get_ohlcv(
    self,
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str = "1d",
    adjust: str = ADJUST_SPLIT,
) -> pd.DataFrame:
    """
    Retrieves OHLCV data for a symbol within a date range.

    `adjust` selects the basis the caller wants — see VALID_ADJUSTMENTS. The
    default reproduces exactly what this method has always returned.
    """
    if adjust not in VALID_ADJUSTMENTS:
        raise ValueError(
            f"Unknown adjustment {adjust!r}; expected one of {VALID_ADJUSTMENTS}"
        )

    query = """
        SELECT date, open, high, low, close, adj_close, volume
        FROM daily_ohlcv
        WHERE symbol = ? AND interval = ? AND date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    with self._get_connection() as conn:
        df = pd.read_sql_query(
            query, conn, params=(symbol, interval, str(start_date), str(end_date))
        )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        # Rename columns to standard YF format for compatibility
        df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        # FORCE NUMERIC TYPES
        # This is critical because if 'Open' contains None (which sqlite returns for NULL),
        # pandas treats the column as 'object', causing interpolation to fail/warn.
        cols_to_numeric = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for c in cols_to_numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        basis = self.get_price_basis([symbol]).get(symbol, BASIS_SPLIT_ADJ)
        if basis != BASIS_SPLIT_ADJ or adjust != ADJUST_SPLIT:
            actions = self.get_actions([symbol]).get(symbol)
            df = self._adjust_frame(df, symbol, basis, adjust, actions)

    return df


def get_ohlcv_batch(
    self,
    symbols: List[str],
    start_date: date,
    end_date: date,
    interval: str = "1d",
    adjust: str = ADJUST_SPLIT,
) -> Dict[str, pd.DataFrame]:
    """
    Retrieves OHLCV data for multiple symbols in a collection of DataFrames.

    `adjust` behaves as in get_ohlcv. Actions and bases are read once for
    the whole batch, and symbols needing no adjustment skip the arithmetic.
    """
    if adjust not in VALID_ADJUSTMENTS:
        raise ValueError(
            f"Unknown adjustment {adjust!r}; expected one of {VALID_ADJUSTMENTS}"
        )
    if not symbols:
        return {}

    placeholders = ", ".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume 
        FROM daily_ohlcv 
        WHERE symbol IN ({placeholders}) AND interval = ? AND date BETWEEN ? AND ?
        ORDER BY symbol, date ASC
    """

    results = {}
    with self._get_connection() as conn:
        # We must pass the list of symbols first, then other params
        params = symbols + [interval, str(start_date), str(end_date)]
        df_all = pd.read_sql_query(query, conn, params=params)

    if not df_all.empty:
        df_all["date"] = pd.to_datetime(df_all["date"])

        bases = self.get_price_basis(symbols)
        # Only symbols that actually need work pay for an actions lookup:
        # a legacy symbol read at the default adjustment is a pure no-op,
        # which is the overwhelmingly common case.
        needs_work = [
            s
            for s in df_all["symbol"].unique()
            if bases.get(s, BASIS_SPLIT_ADJ) != BASIS_SPLIT_ADJ
            or adjust != ADJUST_SPLIT
        ]
        actions_by_symbol = self.get_actions(needs_work) if needs_work else {}

        # Group by symbol and process each
        for sym, group in df_all.groupby("symbol"):
            df = group.drop(columns=["symbol"])
            df.set_index("date", inplace=True)
            df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            # FORCE NUMERIC
            cols_to_numeric = [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            if sym in needs_work:
                df = self._adjust_frame(
                    df,
                    sym,
                    bases.get(sym, BASIS_SPLIT_ADJ),
                    adjust,
                    actions_by_symbol.get(sym),
                )

            results[sym] = df

    return results
