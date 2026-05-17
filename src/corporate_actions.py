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

import numpy as np
import pandas as pd


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
    deduped_splits = deduped_splits.drop(columns=["__split_priority", "__split_ym"])

    # Re-combine, preserving order as much as possible
    result = pd.concat([other_txs, deduped_splits])
    if "original_index" in result.columns:
        result = result.sort_values(by="original_index")
    else:
        result = result.sort_index()
    return result
