# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_analyzer.py
 Purpose:       Contains functions for calculating the current portfolio state,
                holdings, cash balances, and summary metrics from transactions
                and market data. (Refactored from portfolio_logic.py)

 Author:        Kit Matan (Derived from portfolio_logic.py) and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Created:       29/04/2025
 Modified:      2025-04-30
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import logging
from datetime import (
    date,
    datetime,
)  # Used in _build_summary_rows, _process_transactions...

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict

# --- Import Configuration ---
try:
    from config import (
        CASH_SYMBOL_CSV,
        SHORTABLE_SYMBOLS,  # Used by _process_transactions...
        STOCK_QUANTITY_CLOSE_TOLERANCE,  # New tolerance
        # Add any other config constants used by these specific functions if needed
    )
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import from config.py in portfolio_analyzer.py."
    )
    # Define fallbacks if absolutely necessary
    CASH_SYMBOL_CSV = "$CASH"
    SHORTABLE_SYMBOLS = set()
    STOCK_QUANTITY_CLOSE_TOLERANCE = 1e-9  # Fallback if config fails

# --- Import Utilities ---
try:
    from finutils import (
        get_conversion_rate,  # Used by _build_summary_rows
        calculate_irr,  # Used by _build_summary_rows
        get_cash_flows_for_symbol_account,  # Used by _build_summary_rows
        safe_sum,  # Used by _calculate_aggregate_metrics
        get_historical_rate_via_usd_bridge,  # Added for dividend history
        # Add any other finutils functions used by the moved functions
    )
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import from finutils.py in portfolio_analyzer.py."
    )

    # Define dummy functions if needed for structure, but fixing is better
    def get_conversion_rate(*args):
        return np.nan

    def calculate_irr(*args):
        return np.nan

    def get_cash_flows_for_symbol_account(*args):
        return [], []

    def safe_sum(*args):
        return 0.0


from config import _AGGREGATE_CASH_ACCOUNT_NAME_


# --- REVISED: _process_transactions_to_holdings (Split applied to all accounts) ---
@profile
def _process_transactions_to_holdings(
    transactions_df: pd.DataFrame,
    default_currency: str,
    shortable_symbols: Set[str],
    historical_fx_lookup: Dict[Tuple[date, str], float],  # NEW: For historical FX rates
    display_currency_for_hist_fx: str,  # NEW: Target currency for rates in historical_fx_lookup
) -> Tuple[
    Dict[Tuple[str, str], Dict],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Set[int],
    Dict[int, str],
    bool,
]:
    """
    Processes stock/ETF transactions to calculate holdings and aggregate metrics in local currencies.
    (Implementation remains the same as provided previously - relies only on input df and helpers)
    """
    holdings: Dict[Tuple[str, str], Dict] = {}
    overall_realized_gains_local: Dict[str, float] = defaultdict(float)
    overall_dividends_local: Dict[str, float] = defaultdict(float)
    overall_commissions_local: Dict[str, float] = defaultdict(float)
    ignored_row_indices_local = set()
    ignored_reasons_local = {}
    has_warnings = False
    # --- PERFORMANCE OPTIMIZATION ---
    # Auxiliary map to track which accounts hold a given symbol, to avoid iterating
    # over all holdings for every split transaction.
    symbol_to_accounts_map: Dict[str, Set[str]] = defaultdict(set)

    logging.debug(
        "Processing filtered stock/ETF transactions (split logic modified)..."
    )

    required_cols = [
        "Symbol",
        "Account",
        "Type",
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Split Ratio",
        "Local Currency",
        "Date",
        "original_index",
    ]
    missing_cols = [col for col in required_cols if col not in transactions_df.columns]
    if missing_cols:
        logging.error(
            f"CRITICAL ERROR in _process_transactions: Input DataFrame missing required columns: {missing_cols}. Cannot proceed."
        )
        return ({}, {}, {}, {}, ignored_row_indices_local, ignored_reasons_local, True)

    for index, row in transactions_df.iterrows():
        if row["Symbol"] == CASH_SYMBOL_CSV:
            continue

        try:
            original_index = row["original_index"]
            symbol = str(row["Symbol"]).strip()
            account = str(row["Account"]).strip()
            tx_type = str(row["Type"]).lower().strip()
            # --- ADDED: Robust cleaning for local_currency_from_row ---
            local_currency_from_row = str(row["Local Currency"]).strip()
            if (
                not local_currency_from_row
                or local_currency_from_row.upper() in ["<NA>", "NAN", "NONE", "N/A"]
                or len(local_currency_from_row) != 3
            ):
                local_currency_from_row = default_currency  # default_currency is an arg
            else:
                local_currency_from_row = local_currency_from_row.upper()
            # --- END ADDED ---
            tx_date = row["Date"].date()
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
            total_amount_local = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            commission_val = row.get("Commission")
            commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
            commission_local_for_this_tx = (
                0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
            )
            split_ratio = pd.to_numeric(row.get("Split Ratio"), errors="coerce")

            if (
                not symbol
                or not account
                or not tx_type
                or not local_currency_from_row
                or pd.isna(tx_date)
            ):
                raise ValueError(
                    "Essential row data (Symbol, Account, Type, Currency, Date) is missing or invalid."
                )

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            error_msg = f"Row Read Error ({type(e).__name__}): {e}"
            row_repr = row.to_string().replace("\n", " ")[:150]
            logging.warning(
                f"WARN in _process_transactions pre-check row {index} (orig: {row.get('original_index', 'N/A')}): {error_msg}. Data: {row_repr}... Skipping row."
            )
            ignored_reasons_local[row.get("original_index", index)] = error_msg
            ignored_row_indices_local.add(row.get("original_index", index))
            has_warnings = True
            continue

        holding_key_from_row = (symbol, account)
        # --- REMOVED DEBUG FLAG ---
        # log_this_row = account == "E*TRADE"  # Example debug flag
        # --- END REMOVED DEBUG FLAG ---

        if holding_key_from_row not in holdings:
            holdings[holding_key_from_row] = {
                "qty": 0.0,
                "total_cost_local": 0.0,
                "realized_gain_local": 0.0,
                "dividends_local": 0.0,
                "commissions_local": 0.0,
                "local_currency": local_currency_from_row,
                "short_proceeds_local": 0.0,
                "short_original_qty": 0.0,
                "total_cost_invested_local": 0.0,
                "cumulative_investment_local": 0.0,
                "total_buy_cost_local": 0.0,
                "total_cost_display_historical_fx": 0.0,  # NEW: Track cost in display currency at historical FX
            }
            # --- PERFORMANCE OPTIMIZATION ---
            # Track that this account now holds this symbol.
            symbol_to_accounts_map[symbol].add(account)
        elif (
            holdings[holding_key_from_row]["local_currency"] != local_currency_from_row
        ):
            msg = f"Currency mismatch for {symbol}/{account}"
            logging.warning(
                f"CRITICAL WARN in _process_transactions: {msg} row {original_index}. Holding exists with diff ccy. Skip."
            )
            ignored_reasons_local[original_index] = msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            continue

        # --- SPLIT HANDLING ---
        if tx_type in ["split", "stock split"]:
            split_ratio_raw = row.get("Split Ratio")
            logging.debug(
                f"--- SPLIT ROW DEBUG (Row Index: {index}, Orig: {original_index}) ---"
            )
            logging.debug(f"  Symbol: {symbol}, Account: {account}, Date: {tx_date}")
            logging.debug(
                f"  Raw 'Split Ratio' value from row: '{split_ratio_raw}' (Type: {type(split_ratio_raw)})"
            )
            try:
                if pd.isna(split_ratio) or split_ratio <= 0:
                    raise ValueError(f"Invalid split ratio: {split_ratio}")
                logging.debug(
                    f"Processing SPLIT for {symbol} on {tx_date} (Ratio: {split_ratio}). Applying to all accounts holding it."
                )

                affected_accounts = []
                # --- PERFORMANCE OPTIMIZATION ---
                # Instead of iterating all holdings, use the map to find relevant accounts.
                if symbol in symbol_to_accounts_map:
                    affected_accounts = list(symbol_to_accounts_map[symbol])
                    for acc_name in affected_accounts:
                        h_key = (symbol, acc_name)
                        h_data = holdings[h_key]
                        old_qty = h_data["qty"]
                        if abs(old_qty) >= 1e-9:
                            h_data["qty"] *= split_ratio
                            logging.debug(
                                f"  Applied split to {symbol}/{acc_name}: Qty {old_qty:.4f} -> {h_data['qty']:.4f}"
                            )
                            if old_qty < -1e-9 and symbol in shortable_symbols:
                                h_data["short_original_qty"] *= split_ratio
                                if abs(h_data["short_original_qty"]) < 1e-9:
                                    h_data["short_original_qty"] = 0.0
                                logging.debug(
                                    f"  Adjusted short original qty for {symbol}/{acc_name}"
                                )
                            if abs(h_data["qty"]) < 1e-9:
                                h_data["qty"] = 0.0
                        else:
                            logging.debug(
                                f"  Skipped split qty adjust for {symbol}/{acc_name} (Qty near zero: {old_qty:.4f})"
                            )

                logging.debug(
                    f"Split for {symbol} applied to accounts: {affected_accounts}"
                )

                if commission_local_for_this_tx != 0:
                    holding_for_fee = holdings.get(holding_key_from_row)
                    if holding_for_fee:
                        fee_cost = abs(commission_local_for_this_tx)
                        holding_for_fee["commissions_local"] += fee_cost
                        holding_for_fee["total_cost_invested_local"] += fee_cost
                        holding_for_fee["cumulative_investment_local"] += fee_cost
                        overall_commissions_local[local_currency_from_row] += fee_cost
                        logging.debug(
                            f"  Applied split fee {fee_cost:.2f} to specific account {account}"
                        )
                    else:
                        logging.warning(
                            f"  WARN: Could not apply split fee to {holding_key_from_row} - holding not found?"
                        )
                        has_warnings = True
                continue
            except (ValueError, TypeError, KeyError) as e_split:
                error_msg = (
                    f"Split Processing Error ({type(e_split).__name__}): {e_split}"
                )
                logging.warning(
                    f"WARN in _process_transactions SPLIT row {original_index} ({symbol}): {error_msg}. Skipping row."
                )
                ignored_reasons_local[original_index] = error_msg
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                continue
            except Exception as e_split_unexp:
                logging.exception(
                    f"Unexpected error processing SPLIT row {original_index} ({symbol})"
                )
                ignored_reasons_local[original_index] = (
                    "Unexpected Split Processing Error"
                )
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                continue
        # --- END SPLIT HANDLING ---

        holding = holdings.get(holding_key_from_row)
        if not holding:
            logging.error(
                f"CRITICAL LOGIC ERROR: Holding not found for {holding_key_from_row} after initialization. Skipping row {original_index}."
            )
            ignored_reasons_local[original_index] = (
                "Internal Logic Error: Holding not found"
            )
            ignored_row_indices_local.add(original_index)
            # Mark as error? This shouldn't happen. Let's mark warning for now.
            has_warnings = True
            continue

        commission_for_overall = commission_local_for_this_tx
        # --- REMOVED DEBUG FLAG USAGE ---
        # prev_cum_inv = (
        #     holding.get("cumulative_investment_local", 0.0) if log_this_row else 0
        # )
        # --- END REMOVED DEBUG FLAG USAGE ---

        try:
            # --- Validate Numeric Inputs Specific to Transaction Type ---
            if tx_type in [
                "buy",
                "sell",
                "deposit",
                "withdrawal",
                "short sell",
                "buy to cover",
            ]:
                if pd.isna(qty):
                    raise ValueError(f"Missing Quantity for {tx_type}")
                if pd.isna(price_local) and symbol != CASH_SYMBOL_CSV:
                    raise ValueError(f"Missing Price/Share for {tx_type} {symbol}")
            elif tx_type == "dividend":
                if pd.isna(total_amount_local) and pd.isna(price_local):
                    raise ValueError(
                        "Missing both Total Amount and Price/Share for dividend"
                    )
            elif tx_type == "fees":
                if pd.isna(commission_local_raw):
                    raise ValueError("Missing Commission for fees transaction")

            # --- Shorting Logic ---
            if symbol in shortable_symbols and tx_type in [
                "short sell",
                "buy to cover",
            ]:
                qty_abs = abs(qty)
                if qty_abs <= 1e-9:
                    raise ValueError(f"{tx_type} qty must be > 0")
                if tx_type == "short sell":
                    # Cost in display currency at historical FX for short proceeds
                    fx_rate_hist_short = historical_fx_lookup.get(
                        (tx_date, local_currency_from_row), np.nan
                    )
                    if pd.isna(fx_rate_hist_short):
                        logging.warning(
                            f"FX G/L (Short Sell): Missing historical FX for {local_currency_from_row} on {tx_date} for {symbol}. FX G/L on this tx may be inaccurate."
                        )
                        # Fallback: use current rate for this transaction's display currency cost component if historical is missing
                        # This is not ideal but prevents NaN propagation if one rate is missing.
                        # Better: ensure all rates are pre-fetched or handle missing rates by making the whole FX G/L NaN.
                        # For now, let's assume a fallback to 1.0 if display_currency == local_currency, else NaN.
                        fx_rate_hist_short = (
                            1.0
                            if local_currency_from_row == display_currency_for_hist_fx
                            else np.nan
                        )

                    proceeds = (qty_abs * price_local) - commission_local_for_this_tx
                    holding["qty"] -= qty_abs
                    holding["short_proceeds_local"] += proceeds
                    holding["short_original_qty"] += qty_abs
                    holding["commissions_local"] += commission_local_for_this_tx
                    holding["cumulative_investment_local"] -= proceeds
                    if pd.notna(fx_rate_hist_short) and pd.notna(proceeds):
                        holding["total_cost_display_historical_fx"] -= (
                            proceeds * fx_rate_hist_short
                        )  # Proceeds are negative cost

                elif tx_type == "buy to cover":
                    qty_currently_short = (
                        abs(holding["qty"]) if holding["qty"] < -1e-9 else 0.0
                    )
                    if qty_currently_short < 1e-9:
                        raise ValueError(
                            f"Not currently short {symbol}/{account} to cover."
                        )
                    qty_covered = min(qty_abs, qty_currently_short)
                    fx_rate_hist_cover = historical_fx_lookup.get(
                        (tx_date, local_currency_from_row), np.nan
                    )
                    if pd.isna(fx_rate_hist_cover):
                        logging.warning(
                            f"FX G/L (Buy to Cover): Missing historical FX for {local_currency_from_row} on {tx_date} for {symbol}. FX G/L on this tx may be inaccurate."
                        )
                        fx_rate_hist_cover = (
                            1.0
                            if local_currency_from_row == display_currency_for_hist_fx
                            else np.nan
                        )

                    cost = (qty_covered * price_local) + commission_local_for_this_tx
                    if holding["short_original_qty"] <= 1e-9:
                        raise ZeroDivisionError(
                            f"Short original qty is zero/neg for {symbol}/{account}"
                        )
                    avg_proceeds_per_share = (
                        holding["short_proceeds_local"] / holding["short_original_qty"]
                    )
                    proceeds_attributed = qty_covered * avg_proceeds_per_share

                    # Adjust historical display cost for covering
                    if pd.notna(fx_rate_hist_cover) and pd.notna(cost):
                        cost_display_hist_cover = cost * fx_rate_hist_cover
                        # For buy to cover, this cost reduces the "negative cost" (proceeds) accumulated in display currency
                        # Effectively, it's like adding back this cost.
                        # If total_cost_display_historical_fx was negative due to short proceeds, this makes it less negative.
                        holding[
                            "total_cost_display_historical_fx"
                        ] += cost_display_hist_cover

                    gain = proceeds_attributed - cost
                    holding["qty"] += qty_covered
                    holding["short_proceeds_local"] -= proceeds_attributed
                    holding["short_original_qty"] -= qty_covered
                    holding["commissions_local"] += commission_local_for_this_tx
                    holding["realized_gain_local"] += gain
                    overall_realized_gains_local[holding["local_currency"]] += gain
                    if abs(holding["short_original_qty"]) < 1e-9:
                        holding["short_proceeds_local"] = 0.0
                        holding["short_original_qty"] = 0.0
                    if abs(holding["qty"]) < 1e-9:
                        holding["qty"] = 0.0
                    holding["cumulative_investment_local"] += cost
                if commission_for_overall != 0:
                    overall_commissions_local[holding["local_currency"]] += abs(
                        commission_for_overall
                    )
                continue  # Skip standard processing

            # --- Standard Buy/Sell/Dividend/Fee ---
            if tx_type == "buy" or tx_type == "deposit":
                qty_abs = abs(qty)
                if qty_abs <= 1e-9:
                    raise ValueError("Buy/Deposit qty must be > 0")

                fx_rate_hist_buy = historical_fx_lookup.get(
                    (tx_date, local_currency_from_row), np.nan
                )
                if pd.isna(fx_rate_hist_buy):
                    logging.warning(
                        f"FX G/L (Buy): Missing historical FX for {local_currency_from_row} on {tx_date} for {symbol}. FX G/L on this tx may be inaccurate."
                    )
                    fx_rate_hist_buy = (
                        1.0
                        if local_currency_from_row == display_currency_for_hist_fx
                        else np.nan
                    )

                cost = (qty_abs * price_local) + commission_local_for_this_tx
                holding["qty"] += qty_abs
                holding["total_cost_local"] += cost
                holding["commissions_local"] += commission_local_for_this_tx
                holding["total_cost_invested_local"] += cost
                holding["cumulative_investment_local"] += cost
                holding["total_buy_cost_local"] += cost
                if pd.notna(fx_rate_hist_buy) and pd.notna(cost):
                    holding["total_cost_display_historical_fx"] += (
                        cost * fx_rate_hist_buy
                    )

            elif tx_type == "sell" or tx_type == "withdrawal":
                qty_abs = abs(qty)
                held_qty = holding["qty"]
                if held_qty <= 1e-9:
                    msg = f"Sell attempt {symbol}/{account} w/ non-positive long qty ({held_qty:.4f})"
                    logging.warning(
                        f"Warn in _process_transactions: {msg} row {original_index}. Skip."
                    )
                    ignored_reasons_local[original_index] = msg
                    ignored_row_indices_local.add(original_index)
                    has_warnings = True
                    commission_for_overall = 0.0
                    continue
                if qty_abs <= 1e-9:
                    raise ValueError("Sell/Withdrawal qty must be > 0")
                qty_sold = min(qty_abs, held_qty)
                cost_sold = 0.0
                cost_sold_display_historical_fx = (
                    0.0  # For adjusting historical display cost
                )

                if held_qty > 1e-9 and abs(holding["total_cost_local"]) > 1e-9:
                    if pd.isna(holding["total_cost_local"]):
                        cost_sold = 0.0
                        has_warnings = True
                        logging.warning(
                            f"Warning: total_cost_local is NaN for {symbol}/{account} before selling."
                        )
                    else:
                        cost_sold = qty_sold * (holding["total_cost_local"] / held_qty)
                        # Proportionally reduce the historical display cost
                        if pd.notna(holding.get("total_cost_display_historical_fx")):
                            cost_sold_display_historical_fx = qty_sold * (
                                holding["total_cost_display_historical_fx"] / held_qty
                            )

                proceeds = (qty_sold * price_local) - commission_local_for_this_tx
                gain = proceeds - cost_sold
                logging.debug(
                    f"Debug: {symbol}/{account} {qty_sold:.4f} sold at {price_local:.4f}"
                )
                logging.debug(f"Debug: {symbol}/{account} {proceeds:.4f} proceeds")
                logging.debug(f"Debug: {symbol}/{account} {cost_sold:.4f} cost sold")
                logging.debug(f"Debug: {symbol}/{account} {gain:.4f} gain")
                holding["qty"] -= qty_sold
                logging.debug(
                    f"Debug: {symbol}/{account} {holding['qty']:.4f} remaining"
                )
                holding["total_cost_local"] -= cost_sold
                if pd.notna(
                    cost_sold_display_historical_fx
                ):  # Ensure it's a number before subtracting
                    holding[
                        "total_cost_display_historical_fx"
                    ] -= cost_sold_display_historical_fx

                holding["commissions_local"] += commission_local_for_this_tx
                holding["realized_gain_local"] += gain
                overall_realized_gains_local[holding["local_currency"]] += gain
                holding["total_cost_invested_local"] -= cost_sold
                if abs(holding["qty"]) < 1e-9:
                    holding["qty"] = 0.0
                    holding["total_cost_local"] = 0.0
                    holding["total_cost_display_historical_fx"] = (
                        0.0  # Reset if position closed
                    )
                holding["cumulative_investment_local"] -= proceeds

            elif tx_type == "dividend":
                div_amt_local = 0.0
                qty_abs = abs(qty) if pd.notna(qty) else 0
                if pd.notna(total_amount_local) and abs(total_amount_local) > 1e-9:
                    div_amt_local = total_amount_local
                elif pd.notna(price_local) and abs(price_local) > 1e-9:
                    div_amt_local = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                else:
                    div_amt_local = 0.0
                div_effect = (
                    abs(div_amt_local)
                    if (
                        holding.get("qty", 0.0) >= -1e-9
                        or symbol not in shortable_symbols
                    )
                    else -abs(div_amt_local)
                )
                holding["dividends_local"] += div_effect
                overall_dividends_local[holding["local_currency"]] += div_effect
                holding["commissions_local"] += commission_local_for_this_tx
                # Dividends reduce the cost basis in display currency (historical FX)
                fx_rate_hist_div = historical_fx_lookup.get(
                    (tx_date, local_currency_from_row), np.nan
                )
                if pd.isna(fx_rate_hist_div):
                    logging.warning(
                        f"FX G/L (Dividend): Missing historical FX for {local_currency_from_row} on {tx_date} for {symbol}. FX G/L on this tx may be inaccurate."
                    )
                    fx_rate_hist_div = (
                        1.0
                        if local_currency_from_row == display_currency_for_hist_fx
                        else np.nan
                    )
                if pd.notna(fx_rate_hist_div) and pd.notna(div_effect):
                    holding["total_cost_display_historical_fx"] -= (
                        div_effect * fx_rate_hist_div
                    )  # Dividends reduce cost

            elif tx_type == "fees":
                fee_cost = abs(commission_local_for_this_tx)
                holding["commissions_local"] += fee_cost
                holding["total_cost_invested_local"] += fee_cost
                holding["cumulative_investment_local"] += fee_cost
                fx_rate_hist_fee = historical_fx_lookup.get(
                    (tx_date, local_currency_from_row), np.nan
                )
                if pd.isna(fx_rate_hist_fee):
                    fx_rate_hist_fee = (
                        1.0
                        if local_currency_from_row == display_currency_for_hist_fx
                        else np.nan
                    )
                if pd.notna(fx_rate_hist_fee) and pd.notna(fee_cost):
                    holding["total_cost_display_historical_fx"] += (
                        fee_cost * fx_rate_hist_fee
                    )  # Fees add to cost

            else:  # Should not be reachable
                msg = f"Unhandled stock tx type '{tx_type}'"
                logging.warning(
                    f"Warn in _process_transactions: {msg} row {original_index}. Skip."
                )
                ignored_reasons_local[original_index] = msg
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                commission_for_overall = 0.0
                continue

            # --- REMOVED DEBUG FLAG USAGE ---
            # if log_this_row:
            #     ... (logging code removed) ...
            # --- END REMOVED DEBUG FLAG USAGE ---

            if commission_for_overall != 0:
                overall_commissions_local[holding["local_currency"]] += abs(
                    commission_for_overall
                )

        except (ValueError, TypeError, ZeroDivisionError) as e:
            error_msg = f"Calculation Error ({type(e).__name__}): {e}"
            logging.warning(
                f"WARN in _process_transactions row {original_index} ({symbol}, {tx_type}): {error_msg}. Skipping row."
            )
            ignored_reasons_local[original_index] = error_msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            # --- REMOVED DEBUG FLAG USAGE ---
            # if log_this_row:
            #     logging.error(f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}")
            # --- END REMOVED DEBUG FLAG USAGE ---
            continue
        except KeyError as e:
            error_msg = f"Internal Holding Data Error: {e}"
            logging.warning(
                f"WARN in _process_transactions row {original_index} ({symbol}, {tx_type}): {error_msg}. Skipping row."
            )
            ignored_reasons_local[original_index] = error_msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            # --- REMOVED DEBUG FLAG USAGE ---
            # if log_this_row:
            #     logging.error(f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}")
            # --- END REMOVED DEBUG FLAG USAGE ---
            continue
        except Exception as e:
            logging.exception(
                f"Unexpected error processing row {original_index} ({symbol}, {tx_type})"
            )
            ignored_reasons_local[original_index] = "Unexpected Processing Error"
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            # --- REMOVED DEBUG FLAG USAGE ---
            # if log_this_row:
            #     logging.error(f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}")
            # --- END REMOVED DEBUG FLAG USAGE ---
            continue

    # --- Apply STOCK_QUANTITY_CLOSE_TOLERANCE to final holdings ---
    logging.debug(
        f"Applying stock quantity close tolerance of {STOCK_QUANTITY_CLOSE_TOLERANCE}..."
    )
    for holding_key, data in list(
        holdings.items()
    ):  # Use list() for safe iteration if modifying
        symbol, account = holding_key
        if symbol == CASH_SYMBOL_CSV:
            continue  # Skip cash

        current_qty = data.get("qty", 0.0)
        if 0 < abs(current_qty) < STOCK_QUANTITY_CLOSE_TOLERANCE:
            logging.info(
                f"Holding {symbol}/{account} qty {current_qty:.8f} is below tolerance {STOCK_QUANTITY_CLOSE_TOLERANCE}. Setting to 0."
            )
            data["qty"] = 0.0
            data["total_cost_local"] = (
                0.0  # If qty is zero, cost basis should also be zero
            )
            data["total_cost_display_historical_fx"] = (
                0.0  # Also zero out historical display cost
            )
            # Potentially zero out other fields if qty becomes zero, e.g., short proceeds if short position closes due to tolerance
            if data.get("short_original_qty", 0.0) > 0 and data["qty"] == 0.0:
                data["short_proceeds_local"] = 0.0
                data["short_original_qty"] = 0.0
    return (
        holdings,
        dict(overall_realized_gains_local),
        dict(overall_dividends_local),
        dict(overall_commissions_local),
        ignored_row_indices_local,
        ignored_reasons_local,
        has_warnings,
    )


# --- REVISED: _calculate_cash_balances ---
@profile
def _calculate_cash_balances(
    transactions_df: pd.DataFrame, default_currency: str
) -> Tuple[Dict[str, Dict], bool, bool]:
    """
    Calculates the final cash balance, dividends, and commissions for each account's $CASH position.
    (Implementation remains the same as provided previously - relies only on input df and helpers)
    """
    # ... (Function body remains unchanged) ...
    cash_summary: Dict[str, Dict] = {}
    cash_symbol = CASH_SYMBOL_CSV
    has_errors = False
    has_warnings = False

    try:
        cash_transactions = transactions_df[
            transactions_df["Symbol"] == cash_symbol
        ].copy()
        if not cash_transactions.empty:

            def get_signed_quantity_cash(row):
                """Calculates cash flow including commission impact."""  # <-- Corrected Indentation
                type_lower = str(row.get("Type", "")).lower()  # <-- Corrected Indentation
                qty = pd.to_numeric(
                    row.get("Quantity"), errors="coerce"
                )  # <-- Corrected Indentation
                commission_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
                commission = 0.0 if pd.isna(commission_raw) else float(commission_raw)

                return (  # <-- Corrected Indentation
                    0.0
                    if pd.isna(qty)
                    else (
                        # Deposit: Increase cash by quantity MINUS commission
                        abs(qty) - commission
                        if type_lower == "deposit"
                        # Withdrawal: Decrease cash by quantity PLUS commission
                        else (
                            -(abs(qty) + commission)
                            if type_lower == "withdrawal"
                            else 0.0
                        )
                    )
                )

            cash_transactions["SignedQuantity"] = cash_transactions.apply(
                get_signed_quantity_cash, axis=1
            )

            grouped_cash = cash_transactions.groupby("Account")
            cash_qty_agg = grouped_cash["SignedQuantity"].sum()
            cash_comm_agg = grouped_cash["Commission"].sum(min_count=1).fillna(0.0)
            cash_currency_map = grouped_cash["Local Currency"].first()

            cash_dividends_tx = cash_transactions[
                cash_transactions["Type"] == "dividend"
            ].copy()
            cash_div_agg = pd.Series(dtype=float)
            if not cash_dividends_tx.empty:

                def get_dividend_amount(r):
                    total_amt = pd.to_numeric(r.get("Total Amount"), errors="coerce")
                    price = pd.to_numeric(r.get("Price/Share"), errors="coerce")
                    qty = pd.to_numeric(r.get("Quantity"), errors="coerce")
                    qty_abs = abs(qty) if pd.notna(qty) else 0.0
                    if pd.notna(total_amt) and abs(total_amt) > 1e-9:
                        return total_amt
                    elif pd.notna(price) and abs(price) > 1e-9:
                        return (qty_abs * price) if qty_abs > 0 else price
                    else:
                        return 0.0

                cash_dividends_tx["DividendAmount"] = cash_dividends_tx.apply(
                    get_dividend_amount, axis=1
                )
                cash_dividends_tx["Commission"] = pd.to_numeric(
                    cash_dividends_tx["Commission"], errors="coerce"
                ).fillna(0.0)
                cash_dividends_tx["NetDividend"] = (
                    cash_dividends_tx["DividendAmount"]
                    - cash_dividends_tx["Commission"]
                )
                cash_div_agg = cash_dividends_tx.groupby("Account")["NetDividend"].sum()

            all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index).union(
                cash_div_agg.index
            )

            for acc in all_cash_accounts:
                acc_currency = cash_currency_map.get(acc, default_currency)
                acc_balance = (
                    cash_qty_agg.get(acc, 0.0)
                    if isinstance(cash_qty_agg, pd.Series)
                    else (float(cash_qty_agg) if pd.notna(cash_qty_agg) else 0.0)
                )
                acc_commissions = (
                    cash_comm_agg.get(acc, 0.0)
                    if isinstance(cash_comm_agg, pd.Series)
                    else (float(cash_comm_agg) if pd.notna(cash_comm_agg) else 0.0)
                )
                acc_dividends_only = (
                    cash_div_agg.get(acc, 0.0)
                    if isinstance(cash_div_agg, pd.Series)
                    else 0.0
                )
                cash_summary[acc] = {
                    "qty": acc_balance,
                    "realized": 0.0,
                    "dividends": acc_dividends_only,
                    "commissions": acc_commissions,
                    "currency": acc_currency,
                }
        else:
            logging.info(
                "Info in _calculate_cash_balances: No $CASH transactions found."
            )
    except (TypeError, ValueError) as e:
        logging.error(f"Data type/value error calculating cash balances: {e}")
        has_errors = True
    except KeyError as e:
        logging.error(f"Missing expected column calculating cash balances: {e}")
        has_errors = True
    except Exception as e:
        logging.exception(f"Unexpected error calculating cash balances")
        has_errors = True

    return cash_summary, has_errors, has_warnings


# --- REVISED: _build_summary_rows (uses MarketDataProvider) ---
@profile
def _build_summary_rows(
    holdings: Dict[Tuple[str, str], Dict],
    cash_summary: Dict[str, Dict],
    current_stock_data: Dict[
        str, Dict[str, Optional[float]]
    ],  # Data from MarketDataProvider
    current_fx_rates_vs_usd: Dict[str, float],  # Data from MarketDataProvider
    display_currency: str,
    default_currency: str,
    transactions_df: pd.DataFrame,  # Filtered transactions for IRR/fallback
    report_date: date,
    shortable_symbols: Set[str],
    user_excluded_symbols: Set[str],
    user_symbol_map: Dict[str, str],  # New: Accept user symbol map
    manual_prices_dict: Dict[
        str, float
    ],  # This is just the price part of manual_overrides
    include_accounts: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, str], bool, bool]:
    """
    Builds the detailed list of portfolio summary rows, converting values to the display currency.
    (Implementation remains the same as provided previously - relies only on input data and helpers)
    """
    # ... (Function body remains unchanged) ...
    portfolio_summary_rows: List[Dict[str, Any]] = []
    account_market_values_local: Dict[str, float] = defaultdict(float)
    account_local_currency_map: Dict[str, str] = {}
    has_errors = False
    has_warnings = False

    # --- Initialize details for aggregated cash ---
    aggregated_cash_details = {
        "market_value_display": 0.0,
        "dividends_display": 0.0,
        "commissions_display": 0.0,
        # Add other relevant fields if they need to be summed for the cash line
    }

    logging.info(f"Calculating final portfolio summary rows in {display_currency}...")

    # --- Loop 1: Process Stock/ETF Holdings ---
    for holding_key, data in holdings.items():
        symbol, account = holding_key
        current_qty = data.get("qty", 0.0)
        realized_gain_local = data.get("realized_gain_local", 0.0)
        dividends_local = data.get("dividends_local", 0.0)
        commissions_local = data.get("commissions_local", 0.0)
        local_currency = data.get("local_currency", default_currency)
        current_total_cost_local = data.get("total_cost_local", 0.0)
        short_proceeds_local = data.get("short_proceeds_local", 0.0)
        total_cost_invested_local = data.get("total_cost_invested_local", 0.0)
        cumulative_investment_local = data.get("cumulative_investment_local", 0.0)
        total_buy_cost_local = data.get("total_buy_cost_local", 0.0)
        # NEW: Get historical cost in display currency
        total_cost_display_historical_fx_val = data.get(
            "total_cost_display_historical_fx", np.nan
        )

        account_local_currency_map[account] = local_currency
        stock_data = current_stock_data.get(symbol, {})

        # --- Price Determination ---
        price_source = "Unknown"
        current_price_local = np.nan
        day_change_local = np.nan
        day_change_pct = np.nan

        is_excluded = symbol in user_excluded_symbols
        current_price_local_raw = stock_data.get("price")
        is_yahoo_price_valid = (
            pd.notna(current_price_local_raw)
            and isinstance(current_price_local_raw, (int, float))
            and current_price_local_raw > 1e-9
        )

        if not is_excluded and is_yahoo_price_valid:
            price_source = "Yahoo API/Cache"
            current_price_local = float(current_price_local_raw)
            day_change_local_raw = stock_data.get("change")
            day_change_pct_raw = stock_data.get(
                "changesPercentage"
            )  # This is % value from provider
            day_change_local = (
                float(day_change_local_raw)
                if pd.notna(day_change_local_raw)
                else np.nan
            )
            day_change_pct = (
                float(day_change_pct_raw) if pd.notna(day_change_pct_raw) else np.nan
            )
            logging.debug(
                f"Price OK ({symbol}): Using Yahoo price {current_price_local}"
            )
        elif not is_excluded:
            logging.warning(
                f"Warning: Yahoo price invalid/missing for {symbol}. Trying fallbacks."
            )
            price_source = "Yahoo Invalid"
            has_warnings = True
        elif is_excluded:
            logging.debug(
                f"Info: Symbol {symbol} is excluded. Skipping Yahoo fetch, trying fallbacks."
            )
            price_source = "Excluded"
            has_warnings = True

        if pd.isna(current_price_local):
            manual_price = manual_prices_dict.get(symbol)
            if (
                manual_price is not None
                and pd.notna(manual_price)
                and isinstance(manual_price, (int, float))
                and manual_price > 0
            ):
                current_price_local = float(manual_price)
                price_source += " - Manual Fallback"
                logging.debug(
                    f"Info: Used MANUAL fallback price {current_price_local} for {symbol}/{account}"
                )
                day_change_local = np.nan
                day_change_pct = np.nan
            else:
                if symbol in manual_prices_dict:
                    logging.warning(
                        f"Warn: Manual price found for {symbol} but was invalid/zero ({manual_price}). Trying Last TX."
                    )
                else:
                    logging.debug(
                        f"Info: Manual price not found for {symbol}. Trying Last TX."
                    )

        if pd.isna(current_price_local):
            price_source += (
                " - No Manual" if "Manual Fallback" not in price_source else ""
            )
            try:
                fallback_tx = transactions_df[
                    (transactions_df["Symbol"] == symbol)
                    & (transactions_df["Account"] == account)
                    & (transactions_df["Price/Share"].notna())
                    & (
                        pd.to_numeric(transactions_df["Price/Share"], errors="coerce")
                        > 1e-9
                    )
                    & (transactions_df["Date"].dt.date <= report_date)
                ].copy()
                if not fallback_tx.empty:
                    fallback_tx.sort_values(
                        by=["Date", "original_index"], inplace=True, ascending=True
                    )
                    last_tx_row = fallback_tx.iloc[-1]
                    last_tx_price = pd.to_numeric(
                        last_tx_row["Price/Share"], errors="coerce"
                    )
                    last_tx_date = last_tx_row["Date"].date()
                    if pd.notna(last_tx_price) and last_tx_price > 1e-9:
                        current_price_local = float(last_tx_price)
                        price_source += (
                            f" - Last TX ({last_tx_price:.2f}@{last_tx_date})"
                        )
                        logging.debug(
                            f"Info: Used Last TX fallback price {current_price_local} for {symbol}/{account}"
                        )
                    else:
                        logging.warning(
                            f"Warn: Fallback TX found for {symbol}/{account} but price invalid ({last_tx_price}). Using 0."
                        )
                        price_source += " - Last TX Invalid/Zero"
                        current_price_local = 0.0
                else:
                    logging.warning(
                        f"Warn: No valid prior TX found for fallback for {symbol}/{account}. Using 0."
                    )
                    price_source += " - No Last TX"
                    current_price_local = 0.0
            except Exception as e_fallback:
                logging.error(
                    f"ERROR during last TX fallback for {symbol}/{account}: {e_fallback}"
                )
                price_source += " - Fallback Error"
                current_price_local = 0.0
            day_change_local = np.nan
            day_change_pct = np.nan

        if pd.isna(current_price_local):
            logging.error(
                f"ERROR: All price sources failed for {symbol}/{account}. Forcing 0."
            )
            current_price_local = 0.0
            if "Using 0" not in price_source and "Error" not in price_source:
                price_source += " - Forced Zero"
            has_warnings = True
        current_price_local = float(current_price_local)

        # --- Currency Conversion ---
        # Ensure fx_rate is defined before the try block for IRR
        # It's used for converting local currency values to display currency.
        # If this fails, many subsequent calculations will result in NaN.

        fx_rate = get_conversion_rate(
            local_currency, display_currency, current_fx_rates_vs_usd
        )
        if pd.isna(fx_rate):  # Keep critical error log if FX rate is NaN
            logging.error(
                f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for {symbol}/{account}."
            )
            has_errors = True
            fx_rate = np.nan

        # --- Dividend Yield and Income Calculations (in local currency first) ---
        trailing_annual_dividend_rate_local = stock_data.get(
            "trailingAnnualDividendRate"
        )  # This is per share
        dividend_yield_on_current_direct = stock_data.get(
            "dividendYield"
        )  # This is a fraction e.g. 0.02 for 2%

        div_yield_on_cost_pct_local = np.nan
        div_yield_on_current_pct_local = np.nan
        est_annual_income_local = np.nan

        if (
            pd.notna(trailing_annual_dividend_rate_local)
            and trailing_annual_dividend_rate_local > 0
        ):
            # Estimated Annual Income (Local)
            if (
                pd.notna(current_qty) and abs(current_qty) > 1e-9
            ):  # Only for long positions
                est_annual_income_local = (
                    trailing_annual_dividend_rate_local * current_qty
                )

            # Dividend Yield on Cost (Local)
            if (
                pd.notna(current_qty)
                and current_qty > 1e-9
                and pd.notna(current_total_cost_local)
                and current_total_cost_local > 1e-9
            ):
                avg_cost_price_local = current_total_cost_local / current_qty
                if avg_cost_price_local > 1e-9:
                    div_yield_on_cost_pct_local = (
                        trailing_annual_dividend_rate_local / avg_cost_price_local
                    ) * 100.0

            # Dividend Yield on Current Value (Local)
            if (
                pd.notna(dividend_yield_on_current_direct)
                and dividend_yield_on_current_direct > 0
            ):
                div_yield_on_current_pct_local = dividend_yield_on_current_direct
            elif (
                pd.notna(current_price_local) and current_price_local > 1e-9
            ):  # Fallback calculation
                div_yield_on_current_pct_local = (
                    trailing_annual_dividend_rate_local / current_price_local
                )

        # Convert dividend metrics to display currency
        div_yield_on_cost_pct_display = div_yield_on_cost_pct_local  # Yields are percentages, not currency dependent directly once calculated
        div_yield_on_current_pct_display = div_yield_on_current_pct_local

        est_annual_income_display = np.nan
        if pd.notna(est_annual_income_local) and pd.notna(fx_rate):
            est_annual_income_display = est_annual_income_local * fx_rate

        logging.debug(
            f"Symbol: {symbol}, Div Rate Local: {trailing_annual_dividend_rate_local}, Est Income Local: {est_annual_income_local}, Est Income Display: {est_annual_income_display}"
        )
        logging.debug(
            f"  Yield Cost Local: {div_yield_on_cost_pct_local}, Yield Current Local: {div_yield_on_current_pct_local}"
        )

        # --- End Dividend Calculations ---

        # --- ADDED: FX Gain/Loss Calculation ---
        fx_gain_loss_display_holding = np.nan
        fx_gain_loss_pct_holding = np.nan

        # --- Calculate Display Currency Values ---
        market_value_local = current_qty * current_price_local
        if pd.notna(market_value_local):
            account_market_values_local[account] += market_value_local
        market_value_display = (
            market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        day_change_value_display = (
            (current_qty * day_change_local * fx_rate)
            if pd.notna(day_change_local) and pd.notna(fx_rate)
            else np.nan
        )
        current_price_display = (
            current_price_local * fx_rate
            if pd.notna(current_price_local) and pd.notna(fx_rate)
            else np.nan
        )
        cost_basis_display = np.nan
        avg_cost_price_display = np.nan
        unrealized_gain_display = np.nan
        unrealized_gain_pct = np.nan

        is_long = current_qty > 1e-9
        is_short = current_qty < -1e-9
        if is_long:
            cost_basis_display_local = max(0, current_total_cost_local)
            cost_basis_display = (
                cost_basis_display_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(cost_basis_display):
                avg_cost_price_display = (
                    (cost_basis_display / current_qty)
                    if abs(current_qty) > 1e-9
                    else np.nan
                )
                if pd.notna(market_value_display):
                    unrealized_gain_display = market_value_display - cost_basis_display
                    if abs(cost_basis_display) > 1e-9:
                        unrealized_gain_pct = (
                            unrealized_gain_display / cost_basis_display
                        ) * 100.0
                    elif abs(market_value_display) > 1e-9:
                        unrealized_gain_pct = np.inf
                    else:
                        unrealized_gain_pct = 0.0
        elif is_short:
            avg_cost_price_display = np.nan
            cost_basis_display = 0.0
            short_proceeds_display = (
                short_proceeds_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(market_value_display) and pd.notna(short_proceeds_display):
                current_cost_to_cover_display = abs(market_value_display)
                unrealized_gain_display = (
                    short_proceeds_display - current_cost_to_cover_display
                )
                if abs(short_proceeds_display) > 1e-9:
                    unrealized_gain_pct = (
                        unrealized_gain_display / short_proceeds_display
                    ) * 100.0
                elif abs(current_cost_to_cover_display) > 1e-9:
                    unrealized_gain_pct = -np.inf
                else:
                    unrealized_gain_pct = 0.0

        realized_gain_display = (
            realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        dividends_display = dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
        commissions_display = (
            commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        total_cost_invested_display = (
            total_cost_invested_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        cumulative_investment_display = (
            cumulative_investment_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        total_buy_cost_display = (
            total_buy_cost_local * fx_rate if pd.notna(fx_rate) else np.nan
        )

        unrealized_gain_comp = (
            unrealized_gain_display if pd.notna(unrealized_gain_display) else 0.0
        )
        total_gain_display = (
            (
                realized_gain_display
                + unrealized_gain_comp
                + dividends_display
                - commissions_display
            )
            if all(
                pd.notna(v)
                for v in [realized_gain_display, dividends_display, commissions_display]
            )
            else np.nan
        )

        denominator_for_pct = total_buy_cost_display
        total_return_pct = np.nan
        if pd.notna(total_gain_display) and pd.notna(denominator_for_pct):
            if abs(denominator_for_pct) > 1e-9:
                total_return_pct = (total_gain_display / denominator_for_pct) * 100.0
            elif abs(total_gain_display) <= 1e-9:
                total_return_pct = 0.0

        stock_irr = np.nan
        try:
            market_value_local_for_irr = (
                abs(market_value_local)
                if abs(current_qty) > 1e-9 and pd.notna(market_value_local)
                else 0.0
            )
            cf_dates, cf_values = get_cash_flows_for_symbol_account(
                symbol,
                account,
                transactions_df,
                market_value_local_for_irr,
                report_date,
            )
            if cf_dates and cf_values:
                stock_irr = calculate_irr(cf_dates, cf_values)
        except Exception as e_irr:
            logging.warning(
                f"Warning: IRR calculation failed for {symbol}/{account}: {e_irr}",
                exc_info=True,
            )
            has_warnings = True
            stock_irr = np.nan

        # --- FX Gain/Loss Calculation (after all other local values are determined) ---
        fx_gain_loss_display_holding = np.nan  # Initialize
        fx_gain_loss_pct_holding = np.nan  # Initialize

        if local_currency == display_currency:
            fx_gain_loss_display_holding = 0.0
            fx_gain_loss_pct_holding = 0.0  # If no FX G/L, percentage is 0%
        elif (  # Original conditions, but now as elif
            symbol != CASH_SYMBOL_CSV  # Not cash
            and abs(current_total_cost_local) > 1e-9  # Has a local cost basis
            and pd.notna(cost_basis_display)  # Cost basis in display currency is valid
            and abs(cost_basis_display)
            > 1e-9  # Cost basis in display currency is significant
            and pd.notna(
                total_cost_display_historical_fx_val
            )  # Historical cost in display currency is valid
        ):
            try:
                # Cost at current FX: cost_basis_display (which is current_total_cost_local * current_fx_rate)
                # Cost at historical FX: total_cost_display_historical_fx_val
                fx_gain_loss_display_holding = (
                    cost_basis_display - total_cost_display_historical_fx_val
                )

                # Calculate percentage if the historical cost (denominator) is significant
                if (
                    pd.notna(fx_gain_loss_display_holding)
                    and abs(total_cost_display_historical_fx_val)
                    > 1e-9  # Denominator for % is historical cost
                ):
                    fx_gain_loss_pct_holding = (
                        fx_gain_loss_display_holding
                        / total_cost_display_historical_fx_val
                    ) * 100.0
                # If fx_gain_loss_display_holding is calculated as 0 (e.g. rates were identical or very close), pct should also be 0
                elif (
                    pd.notna(fx_gain_loss_display_holding)
                    and abs(fx_gain_loss_display_holding) < 1e-9
                ):
                    fx_gain_loss_pct_holding = 0.0
                # Else, fx_gain_loss_pct_holding remains np.nan if fx_gain_loss_display_holding is nan or historical cost is zero

            except ZeroDivisionError:
                logging.warning(
                    f"FX G/L Calc: ZeroDivisionError for {symbol}/{account}. current_total_cost_local: {current_total_cost_local}"
                )
            except Exception as e_fx_gl:
                logging.error(
                    f"Error calculating FX G/L for {symbol}/{account}: {e_fx_gl}"
                )
        # If none of the above conditions met (e.g. historical cost in display currency is NaN),
        # fx_gain_loss_display_holding and fx_gain_loss_pct_holding will remain np.nan as initialized.

        # --- END FX Gain/Loss Calculation ---

        irr_value_to_store = stock_irr * 100.0 if pd.notna(stock_irr) else np.nan
        if pd.isna(irr_value_to_store) and current_qty != 0:
            logging.debug(f"Debug: IRR is NaN for non-zero holding {symbol}/{account}")

        portfolio_summary_rows.append(
            {
                "Account": account,
                "Symbol": symbol,
                "Quantity": current_qty,
                f"Avg Cost ({display_currency})": avg_cost_price_display,
                f"Price ({display_currency})": current_price_display,
                f"Cost Basis ({display_currency})": cost_basis_display,
                f"Market Value ({display_currency})": market_value_display,
                f"Day Change ({display_currency})": day_change_value_display,
                "Day Change %": day_change_pct,  # Store % value directly
                f"Unreal. Gain ({display_currency})": unrealized_gain_display,
                "Unreal. Gain %": unrealized_gain_pct,
                f"Realized Gain ({display_currency})": realized_gain_display,
                f"Dividends ({display_currency})": dividends_display,
                f"Commissions ({display_currency})": commissions_display,
                f"Total Gain ({display_currency})": total_gain_display,
                f"Total Cost Invested ({display_currency})": total_cost_invested_display,
                "Total Return %": total_return_pct,
                f"Cumulative Investment ({display_currency})": cumulative_investment_display,
                f"Total Buy Cost ({display_currency})": total_buy_cost_display,
                "IRR (%)": irr_value_to_store,
                "Local Currency": local_currency,
                "Price Source": price_source,
                f"Div. Yield (Cost) %": div_yield_on_cost_pct_display,
                f"Div. Yield (Current) %": div_yield_on_current_pct_display,
                f"Est. Ann. Income ({display_currency})": est_annual_income_display,
                f"FX Gain/Loss ({display_currency})": fx_gain_loss_display_holding,
                "FX Gain/Loss %": fx_gain_loss_pct_holding,
            }
        )
    # --- End Stock/ETF Loop ---

    # --- Loop 2: Aggregate CASH Balances (No longer adds individual rows here) ---
    if cash_summary:
        for account, cash_data in cash_summary.items():
            symbol = CASH_SYMBOL_CSV
            current_qty = cash_data.get("qty", 0.0)
            local_currency = cash_data.get("currency", default_currency)
            realized_gain_local = cash_data.get("realized", 0.0)
            dividends_local = cash_data.get("dividends", 0.0)
            commissions_local = cash_data.get("commissions", 0.0)

            # Ensure account's local currency is mapped
            if account not in account_local_currency_map:
                account_local_currency_map[account] = local_currency

            fx_rate = get_conversion_rate(
                local_currency, display_currency, current_fx_rates_vs_usd
            )
            if pd.isna(fx_rate):
                has_errors = True
                fx_rate = np.nan
                logging.error(
                    f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for CASH in {account}."
                )

            market_value_local = current_qty * 1.0
            if pd.notna(market_value_local):
                account_market_values_local[account] = (
                    account_market_values_local.get(account, 0.0) + market_value_local
                )  # Add to existing or initialize

            # Aggregate into display currency
            market_value_display = (
                market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            realized_gain_display = (
                realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            dividends_display = (
                dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            commissions_display_for_acc_cash = (  # Renamed to avoid conflict
                commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
            )

            if pd.notna(market_value_display):
                aggregated_cash_details["market_value_display"] += market_value_display
            if pd.notna(dividends_display):
                aggregated_cash_details["dividends_display"] += dividends_display
            if pd.notna(commissions_display_for_acc_cash):
                aggregated_cash_details[
                    "commissions_display"
                ] += commissions_display_for_acc_cash

    # --- Add a single aggregated cash row if there's any cash ---
    # Only add the aggregated cash row if all accounts are included in the scope.
    if not include_accounts:  # None or empty list implies all accounts
        total_cash_mv_display = aggregated_cash_details["market_value_display"]
        total_cash_div_display = aggregated_cash_details["dividends_display"]
        total_cash_comm_display = aggregated_cash_details["commissions_display"]

        # For cash, total gain is typically dividends minus commissions associated with cash accounts
        total_cash_gain_display = total_cash_div_display - total_cash_comm_display

        # Cost basis, total cost invested, and cumulative investment for cash is its market value,
        # as it represents the net cash position. Total Buy Cost for cash is zero.
        cash_basis_and_investment = total_cash_mv_display

        # Total return for cash itself is usually not calculated this way, or is 0% if based on its own value.
        # If total_cash_gain_display is non-zero and cash_basis_and_investment is non-zero, can calculate.
        total_return_pct_cash = np.nan
        if (
            pd.notna(total_cash_gain_display)
            and pd.notna(cash_basis_and_investment)
            and abs(cash_basis_and_investment) > 1e-9
        ):
            total_return_pct_cash = (
                total_cash_gain_display / cash_basis_and_investment
            ) * 100.0
        elif (
            pd.notna(total_cash_gain_display) and abs(total_cash_gain_display) < 1e-9
        ):  # Zero gain
            total_return_pct_cash = 0.0

        # Add the aggregated cash row
        # Using a special account name defined at the top of the file
        # Using CASH_SYMBOL_CSV for the symbol for internal consistency
        # The display name will be handled by the GUI's PandasModel

        # For cash, price is 1 in its currency. Quantity is the value.
        # Since we are in display_currency, price is 1.0.
        # Quantity is total_cash_mv_display.

        # Avg Cost and Price for aggregated cash in display currency is 1.0
        # if we consider the "Quantity" to be the market value itself.

        portfolio_summary_rows.append(
            {
                "Account": _AGGREGATE_CASH_ACCOUNT_NAME_,
                "Symbol": CASH_SYMBOL_CSV,
                "Quantity": total_cash_mv_display,  # Value is quantity, price is 1
                f"Avg Cost ({display_currency})": (
                    1.0
                    if pd.notna(total_cash_mv_display)
                    and abs(total_cash_mv_display) > 1e-9
                    else np.nan
                ),
                f"Price ({display_currency})": 1.0,  # Price of cash in its currency is 1
                f"Cost Basis ({display_currency})": cash_basis_and_investment,
                f"Market Value ({display_currency})": total_cash_mv_display,
                f"Day Change ({display_currency})": 0.0,  # Cash doesn't change day-to-day by itself
                "Day Change %": 0.0,
                f"Unreal. Gain ({display_currency})": 0.0,  # No unrealized gain on cash
                "Unreal. Gain %": 0.0,
                f"Realized Gain ({display_currency})": 0.0,  # Realized gain is for assets
                f"Dividends ({display_currency})": total_cash_div_display,
                f"Commissions ({display_currency})": total_cash_comm_display,
                f"Total Gain ({display_currency})": total_cash_gain_display,
                f"Total Cost Invested ({display_currency})": cash_basis_and_investment,
                "Total Return %": total_return_pct_cash,
                f"Cumulative Investment ({display_currency})": cash_basis_and_investment,
                f"Total Buy Cost ({display_currency})": 0.0,  # Cash itself has no "buy cost" for portfolio return calculation
                "IRR (%)": np.nan,  # IRR not applicable for cash aggregate this way
                "Local Currency": display_currency,  # Aggregated cash is in display currency
                "Price Source": "N/A (Cash)",
                f"Div. Yield (Cost) %": np.nan,  # Not applicable for cash
                f"Div. Yield (Current) %": np.nan,  # Not applicable for cash
                f"Est. Ann. Income ({display_currency})": np.nan,  # Not applicable for cash
                f"FX Gain/Loss ({display_currency})": 0.0,  # FX G/L for cash is 0
                "FX Gain/Loss %": np.nan,  # Not applicable for cash
            }
        )

    return (
        portfolio_summary_rows,
        dict(account_market_values_local),
        dict(account_local_currency_map),
        has_errors,
        has_warnings,
    )


# --- REVISED: _calculate_aggregate_metrics (Removed transactions_df parameter) ---
@profile
def _calculate_aggregate_metrics(
    full_summary_df: pd.DataFrame,
    display_currency: str,
    # transactions_df: pd.DataFrame, # Parameter removed as unused
    report_date: date,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], bool, bool]:
    """
    Calculates account-level and overall portfolio summary metrics.
    (Implementation remains the same as provided previously - relies only on input df and helpers)
    """
    # ... (Function body remains unchanged, except it no longer references transactions_df) ...
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "mwr": np.nan,
            "total_return_pct": np.nan,
            "total_market_value_display": 0.0,
            "total_realized_gain_display": 0.0,
            "total_unrealized_gain_display": 0.0,
            "total_dividends_display": 0.0,
            "total_commissions_display": 0.0,
            "total_gain_display": 0.0,
            "total_cash_display": 0.0,
            "total_cost_invested_display": 0.0,
            "total_buy_cost_display": 0.0,
            "total_day_change_display": 0.0,
            "total_day_change_percent": np.nan,
        }
    )
    overall_summary_metrics = {}
    has_errors = False
    has_warnings = False

    logging.info("Calculating Account-Level & Overall Metrics...")
    if full_summary_df is None or full_summary_df.empty:
        logging.warning("Input 'full_summary_df' is empty or None.")
        overall_summary_metrics = {
            "market_value": 0.0,
            "cost_basis_held": 0.0,
            "unrealized_gain": 0.0,
            "realized_gain": 0.0,
            "dividends": 0.0,
            "commissions": 0.0,
            "total_gain": 0.0,
            "total_cost_invested": 0.0,
            "total_buy_cost": 0.0,
            "portfolio_mwr": np.nan,
            "day_change_display": 0.0,
            "day_change_percent": np.nan,
            "report_date": report_date.strftime("%Y-%m-%d"),
            "display_currency": display_currency,
            "cumulative_investment": 0.0,
            "total_return_pct": np.nan,
            "est_annual_income_display": 0.0,  # ADDED: Ensure key exists
        }
        return (overall_summary_metrics, dict(account_level_metrics), has_errors, True)

    required_cols = [
        "Account",
        "Symbol",
        "Quantity",
        f"Market Value ({display_currency})",
        f"Total Gain ({display_currency})",
        f"Total Buy Cost ({display_currency})",
    ]
    missing_cols = [col for col in required_cols if col not in full_summary_df.columns]
    if missing_cols:
        logging.warning(
            f"Warning: Aggregate calculation input missing required columns: {missing_cols}. Totals may be inaccurate."
        )
        has_warnings = True

    unique_accounts_in_summary = full_summary_df["Account"].unique()
    for account in unique_accounts_in_summary:
        if account == _AGGREGATE_CASH_ACCOUNT_NAME_:
            continue
        try:
            account_full_df = full_summary_df[full_summary_df["Account"] == account]
            metrics_entry = account_level_metrics[account]
            metrics_entry["mwr"] = np.nan
            cols_to_sum_display = [
                f"Market Value ({display_currency})",
                f"Realized Gain ({display_currency})",
                f"Unreal. Gain ({display_currency})",
                f"Dividends ({display_currency})",
                f"Commissions ({display_currency})",
                f"Total Gain ({display_currency})",
                f"Total Cost Invested ({display_currency})",
                f"Total Buy Cost ({display_currency})",
                f"Day Change ({display_currency})",
                f"FX Gain/Loss ({display_currency})",  # Add FX G/L for aggregation
                # "FX Gain/Loss %" is not directly summed, it's recalculated
            ]
            for col in cols_to_sum_display:
                if col not in account_full_df.columns:
                    logging.warning(
                        f"Warn: Col '{col}' missing for acc '{account}' agg."
                    )
                    has_warnings = True

            metrics_entry["total_market_value_display"] = safe_sum(
                account_full_df, f"Market Value ({display_currency})"
            )
            metrics_entry["total_realized_gain_display"] = safe_sum(
                account_full_df, f"Realized Gain ({display_currency})"
            )
            metrics_entry["total_unrealized_gain_display"] = safe_sum(
                account_full_df, f"Unreal. Gain ({display_currency})"
            )
            metrics_entry["total_dividends_display"] = safe_sum(
                account_full_df, f"Dividends ({display_currency})"
            )
            metrics_entry["total_commissions_display"] = safe_sum(
                account_full_df, f"Commissions ({display_currency})"
            )
            metrics_entry["total_gain_display"] = safe_sum(
                account_full_df, f"Total Gain ({display_currency})"
            )
            cash_mask = (
                account_full_df["Symbol"] == CASH_SYMBOL_CSV
                if "Symbol" in account_full_df.columns
                else pd.Series(False, index=account_full_df.index)
            )
            metrics_entry["total_cash_display"] = (
                safe_sum(
                    account_full_df[cash_mask], f"Market Value ({display_currency})"
                )
                if cash_mask.any()
                else 0.0
            )
            metrics_entry["total_cost_invested_display"] = safe_sum(
                account_full_df, f"Total Cost Invested ({display_currency})"
            )
            acc_total_buy_cost_display = safe_sum(
                account_full_df, f"Total Buy Cost ({display_currency})"
            )
            metrics_entry["total_buy_cost_display"] = acc_total_buy_cost_display
            acc_total_gain = metrics_entry["total_gain_display"]
            acc_denominator = acc_total_buy_cost_display
            acc_total_return_pct = np.nan
            if pd.notna(acc_total_gain) and pd.notna(acc_denominator):
                if abs(acc_denominator) > 1e-9:
                    acc_total_return_pct = (acc_total_gain / acc_denominator) * 100.0
                elif abs(acc_total_gain) <= 1e-9:
                    acc_total_return_pct = 0.0
            metrics_entry["total_return_pct"] = acc_total_return_pct
            acc_total_day_change_display = safe_sum(
                account_full_df, f"Day Change ({display_currency})"
            )
            metrics_entry["total_day_change_display"] = acc_total_day_change_display
            acc_current_mv_display = metrics_entry["total_market_value_display"]
            acc_prev_close_mv_display = np.nan
            if pd.notna(acc_current_mv_display) and pd.notna(
                acc_total_day_change_display
            ):
                acc_prev_close_mv_display = (
                    acc_current_mv_display - acc_total_day_change_display
                )
            metrics_entry["total_day_change_percent"] = np.nan
            if pd.notna(acc_total_day_change_display) and pd.notna(
                acc_prev_close_mv_display
            ):
                if abs(acc_prev_close_mv_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        acc_total_day_change_display / acc_prev_close_mv_display
                    ) * 100.0
                elif abs(acc_total_day_change_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        np.inf if acc_total_day_change_display > 0 else -np.inf
                    )
                elif abs(acc_total_day_change_display) < 1e-9:
                    metrics_entry["total_day_change_percent"] = 0.0

            # --- ADDED: Account-level FX Gain/Loss ---
            acc_fx_gain_loss_display = safe_sum(
                account_full_df, f"FX Gain/Loss ({display_currency})"
            )
            metrics_entry["fx_gain_loss_display"] = acc_fx_gain_loss_display
            acc_cost_basis_display_for_fx_pct = safe_sum(
                account_full_df, f"Cost Basis ({display_currency})"
            )
            metrics_entry["fx_gain_loss_pct"] = (
                (acc_fx_gain_loss_display / acc_cost_basis_display_for_fx_pct) * 100
                if abs(acc_cost_basis_display_for_fx_pct) > 1e-9
                else np.nan
            )
            # --- END ADDED ---

        except Exception as e_acc_agg:
            logging.exception(f"Error aggregating metrics for account '{account}'")
            has_warnings = True

    mkt_val_col = f"Market Value ({display_currency})"
    total_gain_col = f"Total Gain ({display_currency})"
    total_buy_cost_col = f"Total Buy Cost ({display_currency})"
    cost_basis_col = f"Cost Basis ({display_currency})"
    unreal_gain_col = f"Unreal. Gain ({display_currency})"
    real_gain_col = f"Realized Gain ({display_currency})"
    divs_col = f"Dividends ({display_currency})"
    comm_col = f"Commissions ({display_currency})"
    cost_invest_col = f"Total Cost Invested ({display_currency})"
    cum_invest_col = f"Cumulative Investment ({display_currency})"
    day_change_col = f"Day Change ({display_currency})"
    fx_gain_loss_col = f"FX Gain/Loss ({display_currency})"  # For overall sum
    cols_to_check = [
        mkt_val_col,
        total_gain_col,
        total_buy_cost_col,
        cost_basis_col,
        unreal_gain_col,
        real_gain_col,
        divs_col,
        comm_col,
        cost_invest_col,
        cum_invest_col,
        day_change_col,
        fx_gain_loss_col,
    ]
    for col in cols_to_check:
        if col not in full_summary_df.columns:
            logging.warning(f"Warning: Column '{col}' missing for overall aggregation.")
            has_warnings = True

    overall_market_value_display = safe_sum(full_summary_df, mkt_val_col)
    held_mask = pd.Series(False, index=full_summary_df.index)
    if "Quantity" in full_summary_df.columns and "Symbol" in full_summary_df.columns:
        held_mask = (full_summary_df["Quantity"].abs() > 1e-9) | (
            full_summary_df["Symbol"] == CASH_SYMBOL_CSV
        )
    overall_cost_basis_display = (
        safe_sum(full_summary_df.loc[held_mask], cost_basis_col)
        if held_mask.any()
        else 0.0
    )
    overall_unrealized_gain_display = safe_sum(full_summary_df, unreal_gain_col)
    overall_realized_gain_display_agg = safe_sum(full_summary_df, real_gain_col)
    overall_dividends_display_agg = safe_sum(full_summary_df, divs_col)
    overall_commissions_display_agg = safe_sum(full_summary_df, comm_col)
    overall_total_gain_display = safe_sum(full_summary_df, total_gain_col)
    overall_total_cost_invested_display = safe_sum(full_summary_df, cost_invest_col)
    overall_cumulative_investment_display = safe_sum(full_summary_df, cum_invest_col)
    overall_total_buy_cost_display = safe_sum(full_summary_df, total_buy_cost_col)
    overall_day_change_display = safe_sum(full_summary_df, day_change_col)
    overall_prev_close_mv_display = np.nan

    # --- ADDED: Overall FX Gain/Loss ---
    overall_fx_gain_loss_display = safe_sum(full_summary_df, fx_gain_loss_col)
    # --- END ADDED ---

    # --- ADDED: Overall Estimated Annual Income ---
    est_ann_income_col = f"Est. Ann. Income ({display_currency})"
    overall_est_annual_income_display = safe_sum(full_summary_df, est_ann_income_col)
    # --- END ADDED ---

    if pd.notna(overall_market_value_display) and pd.notna(overall_day_change_display):
        overall_prev_close_mv_display = (
            overall_market_value_display - overall_day_change_display
        )
    overall_day_change_percent = np.nan
    if pd.notna(overall_day_change_display) and pd.notna(overall_prev_close_mv_display):
        if abs(overall_prev_close_mv_display) > 1e-9:
            overall_day_change_percent = (
                overall_day_change_display / overall_prev_close_mv_display
            ) * 100.0
        elif abs(overall_day_change_display) > 1e-9:
            overall_day_change_percent = (
                np.inf if overall_day_change_display > 0 else -np.inf
            )
        elif abs(overall_day_change_display) < 1e-9:
            overall_day_change_percent = 0.0

    overall_total_return_pct = np.nan
    overall_denominator = overall_total_buy_cost_display
    if pd.notna(overall_total_gain_display) and pd.notna(overall_denominator):
        if abs(overall_denominator) > 1e-9:
            overall_total_return_pct = (
                overall_total_gain_display / overall_denominator
            ) * 100.0
        elif abs(overall_total_gain_display) <= 1e-9:
            overall_total_return_pct = 0.0

    critical_overall_metrics = [
        overall_market_value_display,
        overall_total_gain_display,
    ]
    if any(pd.isna(m) for m in critical_overall_metrics):
        logging.warning("Warning: Critical overall metrics calculated as NaN.")
        has_warnings = True

    # --- ADDED: Overall FX Gain/Loss Percentage ---
    overall_fx_gain_loss_pct = (
        (overall_fx_gain_loss_display / overall_cost_basis_display) * 100
        if abs(overall_cost_basis_display) > 1e-9
        else np.nan
    )
    # --- END ADDED ---

    overall_summary_metrics = {
        "market_value": overall_market_value_display,
        "cost_basis_held": overall_cost_basis_display,
        "unrealized_gain": overall_unrealized_gain_display,
        "realized_gain": overall_realized_gain_display_agg,
        "dividends": overall_dividends_display_agg,
        "commissions": overall_commissions_display_agg,
        "total_gain": overall_total_gain_display,
        "total_cost_invested": overall_total_cost_invested_display,
        "total_buy_cost": overall_total_buy_cost_display,
        "portfolio_mwr": np.nan,
        "day_change_display": overall_day_change_display,
        "day_change_percent": overall_day_change_percent,
        "report_date": report_date.strftime("%Y-%m-%d"),
        "display_currency": display_currency,
        "cumulative_investment": overall_cumulative_investment_display,
        "total_return_pct": overall_total_return_pct,
        "fx_gain_loss_display": overall_fx_gain_loss_display,  # ADDED
        "fx_gain_loss_pct": overall_fx_gain_loss_pct,  # ADDED
        "est_annual_income_display": overall_est_annual_income_display,  # ADDED
    }
    logging.debug(
        f"--- Finished Aggregating Metrics. Overall Market Value: {overall_market_value_display} ---"
    )

    return (
        overall_summary_metrics,
        dict(account_level_metrics),
        has_errors,
        has_warnings,
    )


# --- NEW: Periodic Return Calculation ---
def calculate_periodic_returns(
    historical_df: pd.DataFrame, benchmark_symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Calculates periodic (Weekly, Monthly, Annual) returns from cumulative gain factors.

    Args:
        historical_df (pd.DataFrame): DataFrame indexed by date, containing cumulative
            gain columns like 'Portfolio Accumulated Gain' and '{Benchmark} Accumulated Gain'.
        benchmark_symbols (List[str]): List of benchmark symbols (e.g., ['SPY', 'QQQ'])
            present in the historical_df columns.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are interval codes ('W', 'M', 'Y')
            and values are DataFrames containing the periodic returns for that interval.
            Returns empty dict if input is invalid.
    """
    # --- ADDED: Log input ---
    logging.debug("--- Entering calculate_periodic_returns ---")
    if not isinstance(historical_df, pd.DataFrame) or historical_df.empty:
        logging.warning(
            "Input historical_df is not a valid DataFrame or is empty. Returning empty dict."
        )
        return {}
    logging.debug(f"Input historical_df shape: {historical_df.shape}")
    logging.debug(f"Input benchmarks: {benchmark_symbols}")
    # --- END ADDED ---

    periodic_returns = {}
    intervals = {"W": "W-FRI", "M": "ME", "Y": "YE"}  # Use Month End, Year End
    if not isinstance(historical_df, pd.DataFrame) or historical_df.empty:
        logging.warning("Cannot calculate periodic returns: Input DataFrame is empty.")
        return periodic_returns

    # Ensure index is DatetimeIndex
    if not isinstance(historical_df.index, pd.DatetimeIndex):
        try:
            historical_df.index = pd.to_datetime(historical_df.index)
        except Exception as e:
            logging.error(f"Error converting index to DatetimeIndex: {e}")
            return periodic_returns

    # Identify relevant columns
    portfolio_col = "Portfolio Accumulated Gain"
    benchmark_cols = [f"{b} Accumulated Gain" for b in benchmark_symbols]
    all_gain_cols = [portfolio_col] + benchmark_cols
    valid_gain_cols = [col for col in all_gain_cols if col in historical_df.columns]

    if not valid_gain_cols:
        logging.warning(
            "Cannot calculate periodic returns: No valid accumulated gain columns found."
        )
        return periodic_returns

    # Define intervals and corresponding pandas frequency codes
    intervals = {"W": "W-FRI", "M": "ME", "Y": "YE"}

    for interval_key, freq_code in intervals.items():
        try:
            # Resample to get the *last* value within each period
            resampled_factors = (
                historical_df[valid_gain_cols].resample(freq_code).last()
            )

            # Calculate period return: (End Factor / Previous End Factor) - 1
            # Use pct_change() which does this calculation efficiently
            period_returns_df = (
                resampled_factors.pct_change() * 100.0
            )  # Multiply by 100 for percentage

            # Rename columns for clarity (optional)
            period_returns_df.columns = [
                col.replace(" Accumulated Gain", f" {interval_key}-Return")
                for col in period_returns_df.columns
            ]

            periodic_returns[interval_key] = period_returns_df.dropna(
                how="all"
            )  # Drop rows where all returns are NaN

        except Exception as e:
            logging.error(f"Error calculating {interval_key} periodic returns: {e}")
            periodic_returns[interval_key] = pd.DataFrame()  # Add empty df on error

    # --- ADDED: Log final result ---
    logging.debug(
        f"--- Exiting calculate_periodic_returns. Result keys: {list(periodic_returns.keys())} ---"
    )
    # --- END ADDED ---
    return periodic_returns


def calculate_correlation_matrix(historical_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix for historical returns of assets.

    Args:
        historical_returns_df (pd.DataFrame): DataFrame where each column represents
                                             an asset's historical returns.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    if historical_returns_df.empty:
        logging.warning(
            "Input historical_returns_df is empty. Cannot calculate correlation matrix."
        )
        return pd.DataFrame()

    # Ensure all columns are numeric
    numeric_df = historical_returns_df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logging.warning(
            "No numeric columns found in historical_returns_df. Cannot calculate correlation matrix."
        )
        return pd.DataFrame()

    correlation_matrix = numeric_df.corr()
    return correlation_matrix


def run_scenario_analysis(
    factor_betas: Dict[str, float],
    scenario_shocks: Dict[str, float],
    portfolio_value: float,
) -> Dict[str, float]:
    """
    Calculates the estimated portfolio impact based on factor betas and scenario shocks.

    Args:
        factor_betas (Dict[str, float]): Dictionary of factor betas (e.g., {'Mkt-RF': 1.2, 'SMB': 0.5}).
        scenario_shocks (Dict[str, float]): Dictionary of factor shocks (e.g., {'Mkt-RF': -0.10}).
        portfolio_value (float): The current market value of the portfolio.

    Returns:
        Dict[str, float]: A dictionary containing the estimated portfolio return and impact.
    """
    estimated_portfolio_return = 0.0
    for factor, shock in scenario_shocks.items():
        beta = factor_betas.get(factor, 0.0)
        estimated_portfolio_return += beta * shock

    estimated_portfolio_impact = portfolio_value * estimated_portfolio_return

    return {
        "estimated_portfolio_return": estimated_portfolio_return,
        "estimated_portfolio_impact": estimated_portfolio_impact,
    }


def extract_dividend_history(
    all_transactions_df: pd.DataFrame,
    display_currency: str,  # <-- Keep existing parameters
    historical_fx_yf: Dict[str, pd.DataFrame],  # YF Ticker -> DataFrame
    default_currency: str,
    include_accounts: Optional[List[str]] = None,  # <-- ADD NEW PARAMETER
) -> pd.DataFrame:
    """
       Extracts dividend history from transactions and converts amounts to the display currency.

       Args:
           all_transactions_df (pd.DataFrame): The full, cleaned transactions DataFrame.
           display_currency (str): The target currency for displaying dividend amounts.
           historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
               (e.g., 'EUR=X') to DataFrames containing historical rates vs USD.
           default_currency (str): The default currency if a transaction's local currency is missing.

    include_accounts (Optional[List[str]], optional): A list of account names to include. If None, includes all accounts. Defaults to None.

       Returns:
           pd.DataFrame: A DataFrame with dividend history, including:
               'Date', 'Symbol', 'Account', 'LocalCurrency', 'DividendAmountLocal',
               'FXRateUsed', 'DividendAmountDisplayCurrency'.
               Returns an empty DataFrame if no dividend transactions or on critical error.
    """
    logging.info(f"Extracting dividend history for display in {display_currency}...")
    logging.debug(
        f"  Input all_transactions_df shape: {all_transactions_df.shape if isinstance(all_transactions_df, pd.DataFrame) else 'Not a DF'}"
    )

    # ADDED LOG: Log include_accounts
    logging.debug(
        f"  extract_dividend_history received include_accounts: {include_accounts}"
    )

    logging.debug(
        f"  Historical FX data keys: {list(historical_fx_yf.keys()) if historical_fx_yf else 'Empty'}"
    )
    if not isinstance(all_transactions_df, pd.DataFrame) or all_transactions_df.empty:
        logging.warning("Dividend history: Input transactions DataFrame is empty.")
        return pd.DataFrame()

    dividend_transactions = all_transactions_df[
        all_transactions_df["Type"].str.lower() == "dividend"
    ].copy()

    # ADDED LOG: Log initial dividend transactions before account filtering
    logging.debug(
        f"  Initial dividend transactions (before account filter): {len(dividend_transactions)} rows."
    )
    if not dividend_transactions.empty:
        logging.debug(
            f"    Accounts in initial dividend transactions: {dividend_transactions['Account'].unique()}"
        )

    # Filter by include_accounts if specified
    if include_accounts is not None and isinstance(include_accounts, list):
        if "Account" in dividend_transactions.columns:
            logging.debug(
                f"  Filtering dividend transactions for accounts: {include_accounts}"
            )
            dividend_transactions = dividend_transactions[
                dividend_transactions["Account"].isin(include_accounts)
            ].copy()
            logging.debug(
                f"  Dividend transactions after account filter: {len(dividend_transactions)} rows."
            )
            if not dividend_transactions.empty:
                logging.debug(
                    f"    Accounts in filtered dividend transactions: {dividend_transactions['Account'].unique()}"
                )
            elif (
                len(
                    all_transactions_df[
                        all_transactions_df["Type"].str.lower() == "dividend"
                    ]
                )
                > 0
            ):  # If initial had dividends but filter resulted in none
                logging.warning(
                    f"    No dividend transactions found for accounts {include_accounts} after filtering. Initial count was {len(all_transactions_df[all_transactions_df['Type'].str.lower() == 'dividend'])}."
                )

    logging.debug(f"  Found {len(dividend_transactions)} dividend transactions.")

    if dividend_transactions.empty:
        logging.info("No dividend transactions found.")
        return pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "Account",
                "LocalCurrency",
                "DividendAmountLocal",
                "FXRateUsed",
                "DividendAmountDisplayCurrency",
            ]
        )  # Return empty DF with schema

    results = []
    for i, (_, row) in enumerate(
        dividend_transactions.iterrows()
    ):  # Added index for logging
        try:
            tx_date = row["Date"].date()  # Convert to date object
            symbol = row["Symbol"]
            account = row["Account"]
            local_curr = row.get("Local Currency", default_currency)

            # Dividend amount in local currency (Net: Total Amount - Commission)
            # For dividends, 'Total Amount' is typically the net received.
            # 'Price/Share' can also be 'Dividend/Share'. 'Quantity' is shares.
            # Let's prioritize 'Total Amount'. If not present, use Qty * Price.
            dividend_local = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            if pd.isna(dividend_local):
                qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
                price_per_share = pd.to_numeric(row.get("Price/Share"), errors="coerce")
                if pd.notna(qty) and pd.notna(price_per_share):
                    dividend_local = qty * price_per_share
                else:
                    logging.warning(
                        f"Could not determine dividend amount for row: {row.get('original_index', 'N/A')}"
                    )
                    continue  # Skip if amount cannot be determined

            commission_local = pd.to_numeric(row.get("Commission"), errors="coerce")
            if pd.notna(commission_local):
                dividend_local -= commission_local  # Subtract commission if present

            fx_rate = get_historical_rate_via_usd_bridge(
                local_curr, display_currency, tx_date, historical_fx_yf
            )
            dividend_display_curr = (
                dividend_local * fx_rate
                if pd.notna(fx_rate) and pd.notna(dividend_local)
                else np.nan
            )

            if i < 5:  # Log first few processed rows
                logging.debug(
                    f"    Processed dividend row (orig_idx: {row.get('original_index', 'N/A')}):"
                )
                logging.debug(
                    f"      Date: {tx_date}, Symbol: {symbol}, Account: {account}, LocalCurr: {local_curr}"
                )
                logging.debug(
                    f"      DividendLocal: {dividend_local}, FXRateUsed: {fx_rate}, DividendDisplayCurr: {dividend_display_curr}"
                )

            results.append(
                [
                    tx_date,
                    symbol,
                    account,
                    local_curr,
                    dividend_local,
                    fx_rate,
                    dividend_display_curr,
                ]
            )
        except Exception as e:
            logging.error(
                f"Error processing dividend row {row.get('original_index', 'N/A')}: {e}"
            )
            continue

    if not results:
        logging.warning("  No results after processing dividend transactions.")
        return pd.DataFrame()

    df_dividends = pd.DataFrame(
        results,
        columns=[
            "Date",
            "Symbol",
            "Account",
            "LocalCurrency",
            "DividendAmountLocal",
            "FXRateUsed",
            "DividendAmountDisplayCurrency",
        ],
    )
    df_dividends.sort_values(by="Date", ascending=False, inplace=True)
    logging.info(f"Extracted {len(df_dividends)} dividend records.")
    if not df_dividends.empty:
        logging.debug(
            f"  Returned df_dividends head:\n{df_dividends.head().to_string()}"
        )
        logging.debug(
            f"  'DividendAmountDisplayCurrency' NaNs: {df_dividends['DividendAmountDisplayCurrency'].isna().sum()} out of {len(df_dividends)}"
        )
    return df_dividends


@profile
def extract_realized_capital_gains_history(
    all_transactions_df: pd.DataFrame,
    display_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],  # YF Ticker -> DataFrame of rates vs USD
    default_currency: str,
    shortable_symbols: Set[
        str
    ],  # Currently unused for CG, but kept for signature consistency
    stock_quantity_close_tolerance: float = STOCK_QUANTITY_CLOSE_TOLERANCE,
    include_accounts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculates realized capital gains from transactions using FIFO accounting for long positions.

    Args:
        all_transactions_df (pd.DataFrame): Cleaned DataFrame of all transactions.
            Expected columns: 'Date', 'Symbol', 'Account', 'Type', 'Local Currency',
                              'Quantity', 'Price/Share', 'Commission', 'Split Ratio', 'original_index'.
        display_currency (str): The target currency for reporting gains.
        historical_fx_yf (Dict[str, pd.DataFrame]): Historical FX rates vs USD, used by
                                                   get_historical_rate_via_usd_bridge.
        default_currency (str): Default currency if 'Local Currency' is missing.
        shortable_symbols (Set[str]): Set of symbols that can be shorted (currently unused for CG).
        stock_quantity_close_tolerance (float): Tolerance for considering a lot closed.
        include_accounts (Optional[List[str]]): Accounts to include. If None, all accounts.

    Returns:
        pd.DataFrame: DataFrame with details of each realized capital gain/loss for long positions.
                      Columns: 'Date', 'Symbol', 'Account', 'Type', 'Quantity',
                               'Avg Sale Price (Local)', 'Total Proceeds (Local)',
                               'Total Cost Basis (Local)', 'Realized Gain (Local)',
                               'Sale/Cover FX Rate', 'Total Proceeds (Display)',
                               'Total Cost Basis (Display)', 'Realized Gain (Display)',
                               'original_tx_id'.
    """
    logging.info(
        f"Extracting realized capital gains history for display in {display_currency}..."
    )

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

    if not isinstance(all_transactions_df, pd.DataFrame) or all_transactions_df.empty:
        logging.warning("Capital gains: Input transactions DataFrame is empty.")
        return pd.DataFrame(columns=output_columns)

    transactions_to_process = all_transactions_df.copy()
    if include_accounts and isinstance(include_accounts, list):
        if "Account" in transactions_to_process.columns:
            transactions_to_process = transactions_to_process[
                transactions_to_process["Account"].isin(include_accounts)
            ]
            logging.info(
                f"Filtered capital gains calculation for accounts: {include_accounts}"
            )
        else:
            logging.warning(
                "Cannot filter by account for capital gains: 'Account' column missing."
            )

    if transactions_to_process.empty:
        logging.info(
            "No transactions to process for capital gains after account filtering."
        )
        return pd.DataFrame(columns=output_columns)

    # Ensure transactions are sorted chronologically
    transactions_to_process.sort_values(by=["Date", "original_index"], inplace=True)

    # Dictionary to store purchase lots: (symbol, account) -> list of lots
    # Each lot: {'qty', 'cost_per_share_local_net', 'purchase_date', 'purchase_fx_to_display', 'original_tx_id'}
    holdings_long: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    realized_gains_records: List[Dict[str, Any]] = []

    for _, row in transactions_to_process.iterrows():
        try:
            tx_date_dt = row["Date"]
            if pd.isna(tx_date_dt):
                logging.warning(
                    f"Skipping CG row due to missing date: {row.get('original_index', 'N/A')}"
                )
                continue
            tx_date = tx_date_dt.date()  # Convert to datetime.date

            symbol = str(row["Symbol"]).strip()
            account = str(row["Account"]).strip()
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
            logging.warning(
                f"Error parsing basic data for CG row {row.get('original_index', 'N/A')}: {e_parse}. Skipping."
            )
            continue

        holding_key = (symbol, account)

        if symbol == CASH_SYMBOL_CSV or tx_type in [
            "deposit",
            "withdrawal",
            "fees",  # Fees on their own don't realize CG on assets
            "dividend",  # Dividends are income, not CG from sale
        ]:
            continue

        if tx_type in ["split", "stock split"]:
            if pd.notna(split_ratio) and split_ratio > 0:
                if holding_key in holdings_long:
                    for lot in holdings_long[holding_key]:
                        lot["qty"] *= split_ratio
                        # Cost per share needs to be adjusted inversely
                        if lot["qty"] > 0:  # Avoid division by zero if qty became 0
                            lot["cost_per_share_local_net"] /= split_ratio
                        else:  # If qty becomes zero due to split (unlikely for positive split_ratio but safety)
                            lot["cost_per_share_local_net"] = 0.0
            continue

        if pd.isna(qty) or qty <= 1e-9:  # Quantity must be positive for buy/sell
            logging.debug(
                f"Skipping CG for {symbol}/{account} on {tx_date} due to zero/NaN quantity: {qty}"
            )
            continue

        # Get FX rate for the current transaction's date (for converting proceeds or cost of new buys)
        fx_rate_to_display_current_tx = get_historical_rate_via_usd_bridge(
            local_curr, display_currency, tx_date, historical_fx_yf
        )
        if pd.isna(fx_rate_to_display_current_tx):
            logging.warning(
                f"CG: Missing FX rate for {local_curr} to {display_currency} on {tx_date} for {symbol} (TX ID: {original_tx_id_current_row}). Display currency gains may be inaccurate for this TX."
            )
            # We will proceed, display currency values will be NaN if this rate is needed and missing.

        if tx_type == "buy":
            if pd.isna(price_local) or price_local <= 1e-9:
                logging.warning(
                    f"Skipping BUY for {symbol}/{account} on {tx_date} (TX ID: {original_tx_id_current_row}) due to invalid price: {price_local}"
                )
                continue

            cost_per_share_local_net = ((qty * price_local) + commission_local) / qty

            holdings_long[holding_key].append(
                {
                    "qty": qty,
                    "cost_per_share_local_net": cost_per_share_local_net,
                    "purchase_date": tx_date,
                    "purchase_fx_to_display": fx_rate_to_display_current_tx,  # FX at time of purchase
                    "original_tx_id": original_tx_id_current_row,  # original_index of the buy transaction
                }
            )

        elif tx_type == "sell":  # Assuming this is selling a long position
            if pd.isna(price_local) or price_local <= 1e-9:
                logging.warning(
                    f"Skipping SELL for {symbol}/{account} on {tx_date} (TX ID: {original_tx_id_current_row}) due to invalid price: {price_local}"
                )
                continue

            qty_to_sell_remaining = qty
            # Proceeds are calculated from the sale price and quantity, less commission on sale
            total_proceeds_local_for_this_sale = (qty * price_local) - commission_local

            total_cost_basis_local_for_this_sale = 0.0
            total_cost_basis_display_for_this_sale = (
                0.0  # Sum of (cost_per_share_local * purchase_fx_rate) * qty_from_lot
            )

            lots_for_holding = holdings_long[holding_key]
            temp_lots_after_sale = []  # Build the new list of lots

            if not lots_for_holding:
                logging.warning(
                    f"Attempted to SELL {qty} of {symbol}/{account} on {tx_date} (TX ID: {original_tx_id_current_row}), but no existing lots found. This might indicate selling short without 'short sell' type or data issue."
                )
                # If we want to strictly prevent this, we can 'continue' here.
                # For now, it will result in zero cost basis if no lots are found.

            for lot in lots_for_holding:
                if (
                    qty_to_sell_remaining <= stock_quantity_close_tolerance
                ):  # Use tolerance
                    temp_lots_after_sale.append(lot)  # Keep unconsumed lot
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
                        total_cost_basis_display_for_this_sale  # Keep existing sum if it's valid
                        if pd.notna(total_cost_basis_display_for_this_sale)
                        else 0.0
                    ) + cost_basis_lot_display_part
                else:  # If any part of cost basis in display is NaN, total becomes NaN
                    total_cost_basis_display_for_this_sale = np.nan

                lot["qty"] -= qty_sold_from_this_lot
                qty_to_sell_remaining -= qty_sold_from_this_lot

                if lot["qty"] >= stock_quantity_close_tolerance:  # Use tolerance
                    temp_lots_after_sale.append(lot)  # Keep partially consumed lot

            holdings_long[holding_key] = (
                temp_lots_after_sale  # Update the lots for the holding
            )

            if (
                qty_to_sell_remaining > stock_quantity_close_tolerance
                and not lots_for_holding
            ):
                # This case was logged above if lots_for_holding was initially empty.
                # If lots_for_holding was not empty but all were consumed and still qty_to_sell_remaining,
                # it means we sold more than was in the tracked lots.
                logging.warning(
                    f"Sold {qty} of {symbol}/{account} on {tx_date} (TX ID: {original_tx_id_current_row}), but only {qty - qty_to_sell_remaining:.4f} shares were covered by existing lots. Remainder to sell: {qty_to_sell_remaining:.4f}."
                )

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

            # Only record if some quantity was actually sold from lots
            quantity_actually_sold_from_lots = qty - qty_to_sell_remaining
            if quantity_actually_sold_from_lots > stock_quantity_close_tolerance:
                realized_gains_records.append(
                    {
                        "Date": tx_date,
                        "Symbol": symbol,
                        "Account": account,
                        "Type": "Sale Long",
                        "Quantity": quantity_actually_sold_from_lots,  # Quantity covered by FIFO lots
                        "Avg Sale Price (Local)": price_local,  # Price per share from sell transaction
                        "Total Proceeds (Local)": total_proceeds_local_for_this_sale
                        * (
                            quantity_actually_sold_from_lots / qty if qty > 0 else 1
                        ),  # Pro-rate proceeds if only partial sale from lots
                        "Total Cost Basis (Local)": total_cost_basis_local_for_this_sale,
                        "Realized Gain (Local)": realized_gain_local
                        * (
                            quantity_actually_sold_from_lots / qty if qty > 0 else 1
                        ),  # Pro-rate gain
                        "Sale/Cover FX Rate": fx_rate_to_display_current_tx,  # FX at time of sale
                        "Total Proceeds (Display)": total_proceeds_display
                        * (
                            quantity_actually_sold_from_lots / qty
                            if qty > 0 and pd.notna(total_proceeds_display)
                            else 1
                        ),
                        "Total Cost Basis (Display)": total_cost_basis_display_for_this_sale,  # This is already sum for sold lots
                        "Realized Gain (Display)": realized_gain_display
                        * (
                            quantity_actually_sold_from_lots / qty
                            if qty > 0 and pd.notna(realized_gain_display)
                            else 1
                        ),
                        "LocalCurrency": local_curr,  # Add the local currency of the sale
                        "original_tx_id": original_tx_id_current_row,  # original_index of the sell transaction
                    }
                )

        # TODO: Implement FIFO for short sale capital gains if required.
        # This would involve tracking short lots: {'qty', 'proceeds_per_share_local_net', 'short_sell_date', 'short_sell_fx_to_display'}
        # and matching 'buy to cover' transactions against these lots.
        # elif tx_type == "short sell":
        #     pass
        # elif tx_type == "buy to cover":
        #     pass

    if not realized_gains_records:
        logging.info(
            "No realized capital gains found after processing all transactions."
        )
        return pd.DataFrame(columns=output_columns)

    df_gains = pd.DataFrame(realized_gains_records, columns=output_columns)
    df_gains.sort_values(by=["Date", "Symbol", "Account"], inplace=True)
    logging.info(f"Extracted {len(df_gains)} realized capital gains records.")
    return df_gains


def calculate_rebalancing_trades(
    holdings_df: pd.DataFrame,
    target_alloc_pct: dict,
    new_cash: float = 0.0,
    display_currency: str = "USD",
):
    """
    Calculates the trades required to rebalance the portfolio to the target allocation.

    Args:
        holdings_df (pd.DataFrame): DataFrame of current holdings.
        target_alloc_pct (dict): Dictionary mapping symbols to their target percentage.
        new_cash (float): New cash to be added or withdrawn from the portfolio.
        display_currency (str): The display currency.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Suggested trades to execute.
            - dict: Summary metrics of the rebalancing plan.
    """
    if holdings_df.empty:
        return pd.DataFrame(), {}

    mkt_val_col = f"Market Value ({display_currency})"
    price_col = f"Price ({display_currency})"

    # Get current cash holding
    current_cash_row = holdings_df[holdings_df["Symbol"] == CASH_SYMBOL_CSV]
    current_cash_value = (
        current_cash_row[mkt_val_col].iloc[0] if not current_cash_row.empty else 0.0
    )

    # Calculate total portfolio value including new cash injection/withdrawal
    # This is the total value we are rebalancing towards
    current_total_market_value_excluding_cash = holdings_df[
        holdings_df["Symbol"] != CASH_SYMBOL_CSV
    ][mkt_val_col].sum()
    total_rebalance_value = (
        current_total_market_value_excluding_cash + current_cash_value + new_cash
    )

    trades = []
    summary = {
        "Total Portfolio Value (After Rebalance)": total_rebalance_value,
        "Total Value to Sell": 0.0,
        "Total Value to Buy": 0.0,
        "Net Cash Change": 0.0,  # This will be calculated based on target cash
        "Estimated Number of Trades": 0,
    }

    # Create a dictionary for quick lookup of current holdings (excluding cash for now)
    current_holdings_values = (
        holdings_df[holdings_df["Symbol"] != CASH_SYMBOL_CSV]
        .set_index("Symbol")[mkt_val_col]
        .to_dict()
    )
    current_holdings_prices = (
        holdings_df[holdings_df["Symbol"] != CASH_SYMBOL_CSV]
        .set_index("Symbol")[price_col]
        .to_dict()
    )
    current_holdings_accounts = (
        holdings_df[holdings_df["Symbol"] != CASH_SYMBOL_CSV]
        .set_index("Symbol")["Account"]
        .to_dict()
    )

    # Process each symbol in the target allocation
    for symbol, target_pct in target_alloc_pct.items():
        target_dollar_value = total_rebalance_value * (target_pct / 100.0)

        if symbol == CASH_SYMBOL_CSV:
            # For CASH, calculate the required cash movement
            cash_movement_needed = target_dollar_value - current_cash_value
            summary["Net Cash Change"] = (
                cash_movement_needed  # This is the final cash adjustment
            )
            # No trade record for CASH itself
        else:
            current_dollar_value = current_holdings_values.get(symbol, 0.0)
            trade_value = target_dollar_value - current_dollar_value

            if (
                abs(trade_value) > 0.01
            ):  # Only consider trades above a certain threshold
                action = "BUY" if trade_value > 0 else "SELL"
                price = current_holdings_prices.get(symbol, 0.0)
                account = current_holdings_accounts.get(symbol, "N/A")

                # If price is 0, quantity will be 0. This is a safe fallback.
                quantity = trade_value / price if price > 0 else 0

                trades.append(
                    {
                        "Action": action,
                        "Symbol": symbol,
                        "Account": account,
                        "Quantity": quantity,
                        "Current Price": price,
                        "Trade Value": trade_value,
                        "Note": f"Rebalance to {target_pct:.2f}%",
                    }
                )

                if action == "BUY":
                    summary["Total Value to Buy"] += trade_value
                else:
                    summary["Total Value to Sell"] += abs(trade_value)

    summary["Estimated Number of Trades"] = len(trades)
    return pd.DataFrame(trades), summary
