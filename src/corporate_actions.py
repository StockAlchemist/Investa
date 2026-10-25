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

import investa_core
import re
from typing import Any, Dict, List, Optional, Tuple

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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_apply_return_of_capital(
        current_qty, current_cost, cash_distributed
    )


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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_apply_stock_dividend(
        current_qty, current_cost, shares_received
    )


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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_apply_spin_off(
        parent_qty, parent_cost, child_qty, allocated_basis
    )


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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_parse_spinoff_description(desc)


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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_build_spinoff_legs(
        parent,
        child,
        child_qty,
        date_str,
        account,
        user_id,
        allocated_basis,
        ratio,
        currency,
    )


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
    # Ported to Rust: rust/investa_core/src/ca.rs
    return investa_core.ca_deduplicate_split_transactions(df)
