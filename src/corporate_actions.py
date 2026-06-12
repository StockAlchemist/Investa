# -*- coding: utf-8 -*-
"""
Corporate-actions helpers extracted from portfolio_logic.py.

This is the foundation module for future corporate-action work (see the
audit recorded in #3 — spin-offs, mergers, return-of-capital, etc.). For
now it only contains split-related helpers; new types will be added as
the supporting transaction types are introduced.

Keeping these as pure DataFrame functions (no class state, no IO) makes
them straightforward to unit-test in isolation.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transaction type registry
# ---------------------------------------------------------------------------
# Types the engine currently understands end-to-end (data loader, all three
# JIT dispatchers in portfolio_logic.py).
SUPPORTED_TYPES = frozenset({
    "buy", "sell",
    "deposit", "withdrawal",
    "dividend", "interest", "tax", "fees",
    "split", "stock split",
    "short sell", "buy to cover",
    "transfer",
})

# Types the engine recognises as valid corporate actions but does NOT yet
# apply mathematically. Rows of these types pass data-loader validation
# (no warning spam) but no holding state is mutated. The math is defined
# below as pure functions; wiring those into the JIT inner loops is the
# remaining work for this epic.
RESERVED_CORPORATE_ACTION_TYPES = frozenset({
    "return of capital",   # apply_return_of_capital
    "stock dividend",      # apply_stock_dividend
    # Multi-symbol actions deferred (need a separate transaction shape):
    "spin off",
    "merger",
})


# ---------------------------------------------------------------------------
# Pure-function arithmetic for the deferred types.
#
# Each function takes the current holding state for a single (symbol, account)
# pair and returns the post-action state. These are intentionally Numba-friendly
# (only floats in, only floats out) so they can be inlined into the JIT
# dispatcher when the integration is wired up. Until then they are exercised
# only by unit tests.
# ---------------------------------------------------------------------------

def apply_return_of_capital(
    current_qty: float,
    current_cost: float,
    cash_distributed: float,
) -> Tuple[float, float, float]:
    """
    A Return-of-Capital distribution returns part of the investor's principal.
    Cost basis is reduced by the cash received; share count is unchanged. If
    the distribution exceeds the remaining basis, the excess becomes a realised
    capital gain (the third return value).

    Returns: (new_qty, new_cost, realised_excess_gain).
    """
    if current_qty <= 1e-9:
        # No position to reduce basis against; the entire distribution is a gain.
        return current_qty, current_cost, cash_distributed

    if cash_distributed <= current_cost:
        return current_qty, current_cost - cash_distributed, 0.0

    # Distribution exceeds basis — basis goes to zero, the remainder is gain.
    excess = cash_distributed - current_cost
    return current_qty, 0.0, excess


def apply_stock_dividend(
    current_qty: float,
    current_cost: float,
    shares_received: float,
) -> Tuple[float, float]:
    """
    A stock dividend distributes additional shares to existing holders without
    requiring payment. The total cost basis is unchanged; the per-share cost
    drops proportionally as share count rises. Equivalent in math to a split
    with ratio = (current_qty + shares_received) / current_qty, but specified
    as an additive share count instead of a multiplier.

    Returns: (new_qty, new_cost).
    """
    if current_qty <= 1e-9 or shares_received <= 1e-9:
        return current_qty, current_cost
    return current_qty + shares_received, current_cost


def deduplicate_split_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate split transactions using a fuzzy-month grouping strategy.

    When a portfolio has both a global "All Accounts" split row AND an
    account-specific row for the same corporate event, the engine would
    multiply quantities twice. This collapses such pairs by keeping the
    "All Accounts" row when present, otherwise the first account-specific
    row by original_index.

    Splits are grouped by (Symbol, Year-Month, Split Ratio) so a single
    event can survive even if recorded across multiple days, while two
    distinct ratios in the same month are kept separate.
    """
    if df is None or df.empty:
        return df

    type_col = "Type"
    if type_col not in df.columns:
        return df

    is_split = df[type_col].str.lower().str.strip().isin(["split", "stock split"])
    if not is_split.any():
        return df

    other_txs = df[~is_split]
    splits_df = df[is_split].copy()

    # Priority: 'All Accounts' (0) > Others (1)
    acc_col = "Account"
    if acc_col in splits_df.columns:
        splits_df["__split_priority"] = np.where(
            splits_df[acc_col].astype(str).str.lower().str.strip() == "all accounts", 0, 1
        )
    else:
        splits_df["__split_priority"] = 1

    # Fuzzy grouping by month
    splits_df["__split_ym"] = pd.to_datetime(splits_df["Date"]).dt.to_period("M")

    # Normalize Split Ratio so 7.0 and 7 don't get treated as different splits.
    splits_df["Split Ratio"] = (
        pd.to_numeric(splits_df["Split Ratio"], errors="coerce").fillna(1.0).astype(float)
    )

    sort_cols = ["Symbol", "__split_ym", "__split_priority"]
    if "original_index" in splits_df.columns:
        sort_cols.append("original_index")
    splits_df = splits_df.sort_values(by=sort_cols)

    # Drop duplicates by Symbol + Month. Ratio included to keep distinct events.
    deduped_splits = splits_df.drop_duplicates(
        subset=["Symbol", "__split_ym", "Split Ratio"], keep="first"
    )
    
    # Avoid .drop() to prevent pandas block manager corruption causing concat IndexError.
    # We explicitly select original columns and make a deep copy.
    deduped_splits = deduped_splits[list(df.columns)].copy()
    other_txs = other_txs.copy()

    # Re-combine, preserving order as much as possible
    frames = [f for f in [other_txs, deduped_splits] if not f.empty]
    if not frames:
        return df.iloc[:0].copy()
    
    if len(frames) == 1:
        result = frames[0].copy()
    else:
        # If concat still fails, fallback to recreating from dicts
        try:
            result = pd.concat(frames)
        except IndexError:
            # Fallback for severe pandas block corruption
            result = pd.DataFrame(pd.concat([
                pd.DataFrame(f.to_dict("list")) for f in frames
            ]))

    if "original_index" in result.columns:
        result = result.sort_values(by="original_index")
    else:
        result = result.sort_index()
    return result
