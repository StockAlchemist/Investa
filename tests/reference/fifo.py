# Frozen reference: calculate_fifo_lots_and_gains from src/portfolio_analyzer.py
# at commit f2688db2, before the Rust port. tests/test_rust_fifo_parity.py holds
# rust/investa_core/src/fifo.rs (and the shim around it) to it, bitwise and
# type for type. Never edit this to follow a change in Rust.
# ruff: noqa
import logging
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import CASH_SYMBOL_CSV, STOCK_QUANTITY_CLOSE_TOLERANCE
from finutils import get_conversion_rate, get_historical_rate_via_usd_bridge


def calculate_fifo_lots_and_gains(
    transactions_df: pd.DataFrame,
    display_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    shortable_symbols: Set[str],
    stock_quantity_close_tolerance: float = STOCK_QUANTITY_CLOSE_TOLERANCE,
    current_fx_rates_vs_usd: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """
    Core FIFO calculation logic.
    Returns:
        - realized_gains_df (pd.DataFrame): The history of realized gains.
        - open_lots (Dict): Dictionary of currently held open lots.
             Key: (Symbol, Account)
             Value: List of Lot Dicts
    """
    output_columns = [
        "Date",
        "Symbol",
        "Account",
        "Type",
        "Quantity",
        "Avg Sale Price (Local)",
        "Total Proceeds (Local)",
        "Total Cost Basis (Local)",
        "Realized Gain (Local)",
        "Sale/Cover FX Rate",
        "Total Proceeds (Display)",
        "Total Cost Basis (Display)",
        "Realized Gain (Display)",
        "LocalCurrency",
        "original_tx_id",
    ]

    # Dictionary to store purchase lots: (symbol, account) -> list of lots
    holdings_long: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    realized_gains_records: List[Dict[str, Any]] = []

    # Track splits already applied to avoid double-processing (symbol, date, ratio)
    processed_splits: Set[Tuple[str, date, float]] = set()

    # Ensure transactions are sorted chronologically
    # Note: We assume transactions_df is already a copy if needed, but safe to sort in place if it's passed as such.
    # To be safe within this function, we operate on the passed df directly if the caller prepared it,
    # or iterate. The caller is responsible for filtering out unwanted rows if any,
    # BUT this function does the main type/symbol filtering.

    # We iterate the DataFrame.
    for _, row in transactions_df.iterrows():
        try:
            tx_date_dt = row["Date"]
            if pd.isna(tx_date_dt):
                continue
            tx_date = tx_date_dt.date()

            symbol = str(row["Symbol"]).upper().strip()
            account = str(row["Account"]).upper().strip()
            tx_type = str(row["Type"]).lower().strip()
            local_curr_raw = row.get("Local Currency")
            local_curr = str(
                local_curr_raw
                if pd.notna(local_curr_raw) and str(local_curr_raw).strip()
                else default_currency
            ).upper()

            if not local_curr or len(local_curr) != 3:
                local_curr = default_currency.upper()

            qty_raw = row.get("Quantity")
            price_local_raw = row.get("Price/Share")
            commission_local_raw = row.get("Commission")
            split_ratio_raw = row.get("Split Ratio")

            qty = pd.to_numeric(qty_raw, errors="coerce")
            price_local = pd.to_numeric(price_local_raw, errors="coerce")
            commission_local = (
                0.0
                if pd.isna(commission_local_raw)
                else float(pd.to_numeric(commission_local_raw, errors="coerce"))
            )
            split_ratio = pd.to_numeric(split_ratio_raw, errors="coerce")
            original_tx_id_current_row = row.get("original_index")

        except Exception as e_parse:
            logging.warning(f"Error parsing basic data for FIFO row: {e_parse}")
            continue

        holding_key = (symbol, account)

        if symbol == CASH_SYMBOL_CSV or tx_type in [
            "deposit",
            "withdrawal",
            "fees",
            "dividend",
        ]:
            continue

        # --- Handle Transfers ---
        if tx_type == "transfer":
            to_account = str(row.get("To Account", "")).upper().strip()
            if not to_account:
                continue

            dest_holding_key = (symbol, to_account)
            qty_to_transfer_remaining = qty
            source_lots = holdings_long[holding_key]
            remaining_source_lots = []
            lots_to_add_to_dest = []

            if not source_lots:
                logging.debug(
                    f"Transfer Info: Transfer {qty} of {symbol} from {account} with no lots tracked (maybe short or data gap)."
                )
                continue

            for lot in source_lots:
                if qty_to_transfer_remaining <= stock_quantity_close_tolerance:
                    remaining_source_lots.append(lot)
                    continue

                qty_taken_from_lot = min(qty_to_transfer_remaining, lot["qty"])

                moved_lot = lot.copy()
                moved_lot["qty"] = qty_taken_from_lot
                lots_to_add_to_dest.append(moved_lot)

                lot["qty"] -= qty_taken_from_lot
                qty_to_transfer_remaining -= qty_taken_from_lot

                if lot["qty"] > stock_quantity_close_tolerance:
                    remaining_source_lots.append(lot)

            holdings_long[holding_key] = remaining_source_lots
            holdings_long[dest_holding_key].extend(lots_to_add_to_dest)
            # Sort by original purchase date/id to maintain FIFO at destination
            holdings_long[dest_holding_key].sort(
                key=lambda x: (x["purchase_date"], x.get("original_tx_id", 0))
            )
            continue

        if tx_type in ["split", "stock split"]:
            if pd.notna(split_ratio) and split_ratio > 0:
                split_event = (symbol, tx_date, float(split_ratio))
                if split_event not in processed_splits:
                    # Global split: apply to all accounts holding this symbol
                    for (h_sym, h_acc), lots in holdings_long.items():
                        if h_sym == symbol:
                            for lot in lots:
                                lot["qty"] *= split_ratio
                                if lot["qty"] > 0:
                                    lot["cost_per_share_local_net"] /= split_ratio
                                else:
                                    lot["cost_per_share_local_net"] = 0.0
                    processed_splits.add(split_event)
            continue

        # --- Spin-off (cash-neutral basis reallocation, two legs by qty) ---
        if tx_type in ["spin off", "spin-off", "spinoff"]:
            total_amt_raw = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            total_amt = abs(float(total_amt_raw)) if pd.notna(total_amt_raw) else 0.0
            if pd.notna(qty) and qty > 1e-9:
                # Child receipt: open a lot at the allocated per-share basis.
                cps = total_amt / qty if qty > 1e-9 else 0.0
                fx_child = get_historical_rate_via_usd_bridge(
                    local_curr, display_currency, tx_date, historical_fx_yf
                )
                holdings_long[holding_key].append(
                    {
                        "qty": qty,
                        "cost_per_share_local_net": cps,
                        "purchase_date": tx_date,
                        "purchase_fx_to_display": fx_child,
                        "original_tx_id": original_tx_id_current_row,
                    }
                )
            else:
                # Parent basis reduction: scale each remaining lot's per-share
                # cost so the total basis drops by `total_amt` (clamped >= 0).
                lots = holdings_long.get(holding_key, [])
                current_basis = 0.0
                for lot in lots:
                    current_basis += lot["qty"] * lot["cost_per_share_local_net"]
                if current_basis > 1e-9:
                    reduce_amt = (
                        total_amt if total_amt < current_basis else current_basis
                    )
                    factor = 1.0 - (reduce_amt / current_basis)
                    for lot in lots:
                        lot["cost_per_share_local_net"] *= factor
            continue

        if pd.isna(qty) or qty <= 1e-9:
            continue

        fx_rate_to_display_current_tx = get_historical_rate_via_usd_bridge(
            local_curr, display_currency, tx_date, historical_fx_yf
        )

        if tx_type == "buy":
            if pd.isna(price_local) or price_local <= 1e-9:
                total_amt_raw = pd.to_numeric(row.get("Total Amount"), errors="coerce")
                if (
                    pd.notna(total_amt_raw)
                    and abs(float(total_amt_raw)) > 1e-9
                    and qty > 1e-9
                ):
                    price_local = abs(float(total_amt_raw)) / qty
                else:
                    continue

            cost_per_share_local_net = ((qty * price_local) + commission_local) / qty

            holdings_long[holding_key].append(
                {
                    "qty": qty,
                    "cost_per_share_local_net": cost_per_share_local_net,
                    "purchase_date": tx_date,
                    "purchase_fx_to_display": fx_rate_to_display_current_tx,
                    "original_tx_id": original_tx_id_current_row,
                }
            )

        elif tx_type == "sell":
            if pd.isna(price_local):
                total_amt_raw = pd.to_numeric(row.get("Total Amount"), errors="coerce")
                if pd.notna(total_amt_raw) and qty > 1e-9:
                    price_local = abs(float(total_amt_raw)) / qty
                else:
                    price_local = 0.0
            price_local = max(0.0, float(price_local))

            qty_to_sell_remaining = qty
            total_proceeds_local_for_this_sale = (qty * price_local) - commission_local
            total_cost_basis_local_for_this_sale = 0.0
            total_cost_basis_display_for_this_sale = 0.0

            lots_for_holding = holdings_long[holding_key]
            temp_lots_after_sale = []

            for lot in lots_for_holding:
                if qty_to_sell_remaining <= stock_quantity_close_tolerance:
                    temp_lots_after_sale.append(lot)
                    continue

                qty_sold_from_this_lot = min(qty_to_sell_remaining, lot["qty"])

                cost_basis_lot_local = (
                    qty_sold_from_this_lot * lot["cost_per_share_local_net"]
                )
                total_cost_basis_local_for_this_sale += cost_basis_lot_local

                cost_basis_lot_display_part = np.nan
                if pd.notna(lot["purchase_fx_to_display"]) and pd.notna(
                    lot["cost_per_share_local_net"]
                ):
                    cost_basis_lot_display_part = (
                        qty_sold_from_this_lot
                        * lot["cost_per_share_local_net"]
                        * lot["purchase_fx_to_display"]
                    )

                if pd.notna(cost_basis_lot_display_part):
                    total_cost_basis_display_for_this_sale = (
                        total_cost_basis_display_for_this_sale
                        if pd.notna(total_cost_basis_display_for_this_sale)
                        else 0.0
                    ) + cost_basis_lot_display_part
                else:
                    total_cost_basis_display_for_this_sale = np.nan

                lot["qty"] -= qty_sold_from_this_lot
                qty_to_sell_remaining -= qty_sold_from_this_lot

                if lot["qty"] >= stock_quantity_close_tolerance:
                    temp_lots_after_sale.append(lot)

            holdings_long[holding_key] = temp_lots_after_sale

            realized_gain_local = (
                total_proceeds_local_for_this_sale
                - total_cost_basis_local_for_this_sale
            )

            total_proceeds_display = np.nan
            if pd.notna(fx_rate_to_display_current_tx) and pd.notna(
                total_proceeds_local_for_this_sale
            ):
                total_proceeds_display = (
                    total_proceeds_local_for_this_sale * fx_rate_to_display_current_tx
                )

            realized_gain_display = np.nan
            if pd.notna(total_proceeds_display) and pd.notna(
                total_cost_basis_display_for_this_sale
            ):
                realized_gain_display = (
                    total_proceeds_display - total_cost_basis_display_for_this_sale
                )

            # Fallback logic for display gain
            if pd.isna(realized_gain_display) and pd.notna(realized_gain_local):
                if current_fx_rates_vs_usd:
                    try:
                        fallback_rate = get_conversion_rate(
                            local_curr, display_currency, current_fx_rates_vs_usd
                        )
                        if pd.notna(fallback_rate):
                            realized_gain_display = realized_gain_local * fallback_rate
                    except Exception:
                        pass
                if pd.isna(realized_gain_display) and local_curr == display_currency:
                    realized_gain_display = realized_gain_local

            realized_gains_records.append(
                {
                    "Date": tx_date,
                    "Symbol": symbol,
                    "Account": account,
                    "Type": "Sale Long",
                    "Quantity": qty,
                    "Avg Sale Price (Local)": price_local,
                    "Total Proceeds (Local)": total_proceeds_local_for_this_sale,
                    "Total Cost Basis (Local)": total_cost_basis_local_for_this_sale,
                    "Realized Gain (Local)": realized_gain_local,
                    "Sale/Cover FX Rate": fx_rate_to_display_current_tx,
                    "Total Proceeds (Display)": total_proceeds_display,
                    "Total Cost Basis (Display)": total_cost_basis_display_for_this_sale,
                    "Realized Gain (Display)": realized_gain_display,
                    "LocalCurrency": local_curr,
                    "original_tx_id": original_tx_id_current_row,
                }
            )

    if not realized_gains_records:
        return pd.DataFrame(columns=output_columns), dict(holdings_long)

    df_gains = pd.DataFrame(realized_gains_records, columns=output_columns)
    df_gains.sort_values(by=["Date", "Symbol", "Account"], inplace=True)

    return df_gains, dict(holdings_long)
