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

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transaction type registry
# ---------------------------------------------------------------------------
# Types the engine currently understands end-to-end (data loader, all three
# JIT dispatchers in portfolio_logic.py).
SUPPORTED_TYPES = frozenset(
    {
        "buy",
        "sell",
        "deposit",
        "withdrawal",
        "dividend",
        "interest",
        "tax",
        "fees",
        "split",
        "stock split",
        "short sell",
        "buy to cover",
        "transfer",
        # Spin-off is applied end-to-end by the JIT holdings dispatchers. It is
        # imported as two "spin off" legs sharing a date (see apply_spin_off):
        #   - child receipt  (Quantity > 0): adds the new symbol at its allocated
        #     cost basis, cash-neutral.
        #   - parent basis reduction (Quantity == 0): reduces the parent symbol's
        #     cost basis by the amount allocated to the child, cash-neutral.
        "spin off",
    }
)

# Types the engine recognises as valid corporate actions but does NOT yet
# apply mathematically. Rows of these types pass data-loader validation
# (no warning spam) but no holding state is mutated. The math is defined
# below as pure functions; wiring those into the JIT inner loops is the
# remaining work for this epic.
RESERVED_CORPORATE_ACTION_TYPES = frozenset(
    {
        "return of capital",  # apply_return_of_capital
        "stock dividend",  # apply_stock_dividend
        # Multi-symbol action still deferred (needs a separate transaction shape):
        "merger",
    }
)


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


def apply_spin_off(
    parent_qty: float,
    parent_cost: float,
    child_qty: float,
    allocated_basis: float,
) -> Tuple[float, float, float, float]:
    """
    A spin-off distributes shares of a newly independent company to the holders
    of the parent. It is economically cash-neutral: no money changes hands, and
    the parent's cost basis is split between the parent and the child according
    to their relative fair-market values (the broker performs this allocation —
    e.g. IBKR reports the child's allocated basis in its Open Positions table).

    The parent keeps its share count; only its cost basis is reduced by the
    amount allocated to the child. The child position is created with that same
    amount as its cost basis. Total basis across the two symbols is preserved.

    ``allocated_basis`` is clamped to the parent's remaining basis so a stale or
    over-large allocation can never drive the parent basis negative (the excess
    is simply not moved).

    Returns: (new_parent_qty, new_parent_cost, new_child_qty, new_child_cost).
    """
    moved = allocated_basis
    if moved < 0.0:
        moved = 0.0
    if moved > parent_cost:
        moved = parent_cost
    return parent_qty, parent_cost - moved, child_qty, moved


# ---------------------------------------------------------------------------
# Shared spin-off parsing / row construction.
#
# Both IBKR import paths (the Activity-Statement PDF parser and the Flex-XML
# connector) describe a spin-off with the same free-text description, e.g.
#   "SPGI(US78409V1044) Spinoff 1 for 1 (MBGL, MOBILITY GLOBAL INC, US60744M1062)"
# Centralising the description parsing and the two-leg row construction here
# keeps the importers in lock-step and unit-testable without a broker fixture.
# ---------------------------------------------------------------------------

_SPINOFF_PARENT_RE = re.compile(r"^\s*([A-Z0-9.]{1,6})\s*\(")
_SPINOFF_PAREN_GROUP_RE = re.compile(r"\(([^)]*)\)")
_SPINOFF_RATIO_RE = re.compile(r"(\d+\s+for\s+\d+)", re.IGNORECASE)


def parse_spinoff_description(desc: str) -> Optional[Tuple[str, str, str]]:
    """Extract (parent_symbol, child_symbol, ratio) from an IBKR spin-off
    description, or None if it isn't a spin-off / can't be parsed.

    The child symbol is the first token of the final "(TICKER, NAME, ISIN)"
    group; the parent is the leading token before its CUSIP parenthesis.
    """
    if not desc:
        return None
    low = desc.lower().replace("-", " ")
    if "spin" not in low or "off" not in low:
        return None

    m_parent = _SPINOFF_PARENT_RE.match(desc)
    if not m_parent:
        return None
    parent = m_parent.group(1)

    child = None
    for group in reversed(_SPINOFF_PAREN_GROUP_RE.findall(desc)):
        first = group.split(",")[0].strip()
        if re.fullmatch(r"[A-Z0-9.]{1,6}", first):
            child = first
            break
    if not child or child == parent:
        return None

    ratio_m = _SPINOFF_RATIO_RE.search(desc)
    ratio = ratio_m.group(1) if ratio_m else "spin-off"
    return parent, child, ratio


def build_spinoff_legs(
    parent: str,
    child: str,
    child_qty: float,
    date_str: str,
    account: str,
    user_id: int,
    allocated_basis: float,
    ratio: str = "spin-off",
    currency: str = "USD",
) -> List[Dict[str, Any]]:
    """Build the transaction rows a spin-off decomposes into for the engine:

      1. child receipt        (Quantity > 0): creates the new position at its
         allocated cost basis, cash-neutral.
      2. parent basis cut      (Quantity == 0): reduces the parent's cost basis
         by the amount moved to the child. Emitted only when the allocation is
         known (> 0) — a zero reduction is a no-op and just adds ledger noise.

    The JIT holdings dispatchers distinguish the two legs by Quantity. If the
    allocation is unknown (0), only the child receipt is emitted so the received
    shares are never silently dropped.
    """
    if child_qty <= 1e-9:
        return []
    allocated = float(allocated_basis or 0.0)
    if allocated < 0.0:
        allocated = 0.0
    per_share = (allocated / child_qty) if child_qty > 1e-9 else 0.0

    legs: List[Dict[str, Any]] = [
        {
            "Date": date_str,
            "Type": "Spin-off",
            "Symbol": child,
            "Quantity": child_qty,
            "Price/Share": per_share,
            "Total Amount": allocated,
            "Commission": 0.0,
            "Account": account,
            "Note": f"Spinoff {ratio} from {parent}"[:100],
            "Local Currency": currency,
            "user_id": user_id,
        }
    ]
    if allocated > 1e-9:
        legs.append(
            {
                "Date": date_str,
                "Type": "Spin-off",
                "Symbol": parent,
                "Quantity": 0.0,
                "Price/Share": 0.0,
                "Total Amount": allocated,
                "Commission": 0.0,
                "Account": account,
                "Note": f"Spinoff basis reallocation to {child}"[:100],
                "Local Currency": currency,
                "user_id": user_id,
            }
        )
    return legs


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

    # Safely de-fragment the DataFrame before ANY slicing or copying.
    # The incoming DataFrame often arrives with "Gaps in blk ref_locs" block manager
    # corruption from upstream concatenations, which causes ~is_split or .copy() to crash.
    try:
        df = pd.DataFrame({c: df[c].to_numpy() for c in df.columns}, index=df.index)
    except Exception:
        original_index = df.index
        df = pd.DataFrame(df.to_dict("records"))
        df.index = original_index

    other_txs = df[~is_split]
    splits_df = df[is_split].copy()

    # Priority: 'All Accounts' (0) > Others (1)
    acc_col = "Account"
    if acc_col in splits_df.columns:
        splits_df["__split_priority"] = np.where(
            splits_df[acc_col].astype(str).str.lower().str.strip() == "all accounts",
            0,
            1,
        )
    else:
        splits_df["__split_priority"] = 1

    # Fuzzy grouping by month
    splits_df["__split_ym"] = pd.to_datetime(splits_df["Date"]).dt.to_period("M")

    # Normalize Split Ratio so 7.0 and 7 don't get treated as different splits.
    splits_df["Split Ratio"] = (
        pd.to_numeric(splits_df["Split Ratio"], errors="coerce")
        .fillna(1.0)
        .astype(float)
    )

    sort_cols = ["Symbol", "__split_ym", "__split_priority"]
    if "original_index" in splits_df.columns:
        sort_cols.append("original_index")
    splits_df = splits_df.sort_values(by=sort_cols)

    # Drop duplicates by Symbol + Month. Ratio included to keep distinct events.
    deduped_splits = splits_df.drop_duplicates(
        subset=["Symbol", "__split_ym", "Split Ratio"], keep="first"
    )

    # We must explicitly select original columns and make a deep copy, since
    # dropping columns can again cause block manager issues during concat.
    deduped_splits = deduped_splits[list(df.columns)].copy()
    other_txs = other_txs.copy()

    # Re-combine, preserving order as much as possible
    frames = [f for f in [other_txs, deduped_splits] if not f.empty]
    if not frames:
        return df.iloc[:0].copy()

    if len(frames) == 1:
        result = frames[0].copy()
    else:
        result = pd.concat(frames)

    if "original_index" in result.columns:
        result = result.sort_values(by="original_index")
    else:
        result = result.sort_index()
    return result
