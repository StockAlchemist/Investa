"""Numba-JIT valuation kernels and Python fallbacks for holdings/portfolio value.

Split from portfolio_logic.py. Code inside the @numba.jit functions must stay
NumPy-compatible (nopython mode): no Python objects, NumPy arrays only.
"""

# ruff: noqa: E402
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set, Tuple

import numba
import numpy as np
import pandas as pd

from config import (
    CASH_SYMBOL_CSV,
    HISTORICAL_CALC_METHOD,
    HISTORICAL_DEBUG_DATE_VALUE,
    SHORTABLE_SYMBOLS,
    STOCK_QUANTITY_CLOSE_TOLERANCE,
    YFINANCE_EXCLUDED_SYMBOLS,
)
from finutils import get_historical_price, get_historical_rate_via_usd_bridge

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


def _normalize_series(series):
    """Normalizes a pandas Series to uppercase and trimmed strings."""
    if series.empty:
        return series
    return series.astype(str).str.upper().str.strip()


def _calculate_portfolio_value_at_date_unadjusted_python(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    processed_warnings: set,
    included_accounts: Optional[List[str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices (Pure Python version).

    This function simulates the portfolio state up to the `target_date` by processing
    all transactions (buys, sells, splits, shorts, cash movements) chronologically.
    It determines the quantity of each holding (stocks and cash) per account.
    It then uses the derived *unadjusted* historical stock prices and historical FX rates
    for the `target_date` to calculate the market value of each position in the
    `target_currency`. Cash is valued at 1.0 in its local currency. Includes fallback
    logic using the last known transaction price if historical price is unavailable.

    Args:
        target_date (date): The date for which to calculate the portfolio value.
        transactions_df (pd.DataFrame): DataFrame containing all cleaned transactions.
        historical_prices_yf_unadjusted (Dict[str, pd.DataFrame]): Dictionary mapping YF tickers
            to DataFrames containing derived *unadjusted* historical prices.
        historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
            to DataFrames containing historical rates vs USD.
        target_currency (str): The currency code for the output portfolio value.
        internal_to_yf_map (Dict[str, str]): Dictionary mapping internal symbols to YF tickers.
        account_currency_map (Dict[str, str]): Mapping of account names to their local currencies.
        default_currency (str): Default currency if not found.
        manual_overrides_dict (Optional[Dict[str, Dict[str, Any]]]): Manual overrides for price, etc.
        processed_warnings (set): A set used to track and avoid logging duplicate warnings.

    Returns:
        Tuple[float, bool]:
            - total_market_value_display_curr_agg (float): The total portfolio market value
                in the target currency for the date. Returns np.nan if any critical price/FX lookup fails.
            - any_lookup_nan_on_date (bool): True if any required price or FX rate lookup failed critically.
    """
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )
    if IS_DEBUG_DATE:
        logging.debug(f"--- DEBUG VALUE CALC for {target_date} ---")
        logging.debug(f"  Target Currency: {target_currency}")
        logging.debug(f"  Included Accounts: {included_accounts}")
        logging.debug(f"  Transactions count up to date: {len(transactions_df[transactions_df['Date'].dt.date <= target_date])}")

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        if IS_DEBUG_DATE:
            logging.debug(f"  No transactions found up to {target_date}.")
        return 0.0, False

    # --- ADDED: Normalize included_accounts ---
    included_accounts_norm = set()
    if included_accounts:
        included_accounts_norm = {acc.strip().upper() for acc in included_accounts}
    # --- END ADDED ---

    # --- ADDED: Track last known prices for fallback ---
    last_known_prices: Dict[Tuple[str, str], float] = {}
    # --- END ADDED ---

    holdings: Dict[Tuple[str, str], Dict] = {}
    processed_splits: Set[Tuple[str, date, float]] = set()
    for index, row in transactions_til_date.iterrows():
        symbol = str(row.get("Symbol", "UNKNOWN")).strip()
        # Normalize account
        account_raw = str(row.get("Account", "Unknown"))
        account = account_raw.strip().upper()
        local_currency_from_row = str(row.get("Local Currency", default_currency))
        holding_key_from_row = (symbol, account)
        tx_type = str(row.get("Type", "UNKNOWN_TYPE")).lower().strip()
        tx_date_row = row["Date"].date()

        # Update last known price ONLY for transaction types where Price/Share is
        # the stock price. Dividend / Tax / Interest / Fee rows store the
        # per-share dividend or fee amount in that column, which would poison
        # the last-known-price fallback and create huge phantom valuation jumps
        # on days where yfinance returns NaN for that symbol.
        if tx_type in ("buy", "sell", "short sell", "buy to cover", "transfer"):
            try:
                tx_price = pd.to_numeric(row.get("Price/Share"), errors="coerce")
                if pd.notna(tx_price) and tx_price > 1e-9:
                    last_known_prices[holding_key_from_row] = float(tx_price)
            except Exception:
                pass

        if symbol != CASH_SYMBOL_CSV and holding_key_from_row not in holdings:
            holdings[holding_key_from_row] = {
                "qty": 0.0,
                "local_currency": local_currency_from_row,
                "is_stock": True,
            }
        elif (
            symbol != CASH_SYMBOL_CSV
            and holdings[holding_key_from_row]["local_currency"]
            != local_currency_from_row
        ):
            holdings[holding_key_from_row]["local_currency"] = local_currency_from_row
            if IS_DEBUG_DATE:
                logging.debug(
                    f"  WARN (Value Calc): Currency overwritten for {holding_key_from_row} to {local_currency_from_row}"
                )

        if symbol == CASH_SYMBOL_CSV:
            continue

        try:
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            split_ratio = pd.to_numeric(row.get("Split Ratio"), errors="coerce")
            if tx_type in ["split", "stock split"]:
                if pd.notna(split_ratio) and split_ratio > 0:
                    split_event = (symbol, tx_date_row, float(split_ratio))
                    if split_event not in processed_splits:
                        for h_key, h_data in holdings.items():
                            h_symbol, _ = h_key
                            if h_symbol == symbol:
                                old_qty = h_data["qty"]
                                if abs(old_qty) >= 1e-9:
                                    h_data["qty"] *= split_ratio
                                    if IS_DEBUG_DATE:
                                        logging.debug(
                                            f"  Applying global split ratio {split_ratio} to {h_key} (Date: {tx_date_row}) Qty: {old_qty:.4f} -> {h_data['qty']:.4f}"
                                        )
                                    if abs(h_data["qty"]) < 1e-9:
                                        h_data["qty"] = 0.0
                        processed_splits.add(split_event)
                else:
                    if IS_DEBUG_DATE:
                        logging.warning(
                            f"  Skipping invalid split ratio ({split_ratio}) for {symbol} on {tx_date_row}"
                        )
                continue

            holding_to_update = holdings.get(holding_key_from_row)
            if not holding_to_update:
                continue

            if symbol in SHORTABLE_SYMBOLS and tx_type in [
                "short sell",
                "buy to cover",
            ]:
                if pd.isna(qty):
                    continue
                qty_abs = abs(qty)
                if tx_type == "short sell":
                    holding_to_update["qty"] -= qty_abs
                elif tx_type == "buy to cover":
                    current_short_qty_abs = (
                        abs(holding_to_update["qty"])
                        if holding_to_update["qty"] < -1e-9
                        else 0.0
                    )
                    qty_being_covered = min(qty_abs, current_short_qty_abs)
                    holding_to_update["qty"] += qty_being_covered
            elif tx_type == "buy" or tx_type == "deposit":
                if pd.notna(qty) and qty > 0:
                    holding_to_update["qty"] += qty
            elif tx_type == "sell" or tx_type == "withdrawal":
                if pd.notna(qty) and qty > 0:
                    sell_qty = qty
                    held_qty = holding_to_update["qty"]
                    qty_sold = min(sell_qty, held_qty) if held_qty > 1e-9 else 0
                    holding_to_update["qty"] -= qty_sold
            elif tx_type == "transfer":
                to_account_raw = str(row.get("To Account", ""))
                to_account = to_account_raw.strip().upper()
                if pd.notna(qty) and qty > 0:
                    transfer_qty = qty

                    # 1. Deduct from Source Account
                    holding_to_update["qty"] -= transfer_qty

                    # 2. Add to Destination Account
                    if to_account and transfer_qty > 0:
                        to_key = (symbol, to_account)
                        if to_key not in holdings:
                            holdings[to_key] = {
                                "qty": 0.0,
                                "local_currency": local_currency_from_row,
                                "is_stock": True,
                            }
                        holdings[to_key]["qty"] += transfer_qty

                        # --- ADDED: Propagate last known price to destination ---
                        if holding_key_from_row in last_known_prices:
                            last_known_prices[to_key] = last_known_prices[holding_key_from_row]
                        # --- END ADDED ---

                        # This function only calculates market value, not cost basis,
                        # so only the quantity needs to be moved. The Numba version
                        # below is where the cost basis logic is critical.
                        if IS_DEBUG_DATE:
                            logging.debug(
                                f"  Transferring {transfer_qty} of {symbol} from {account} to {to_account}"
                            )
                            logging.debug(
                                f"    New Source Qty: {holding_to_update['qty']:.4f}"
                            )
                            logging.debug(
                                f"    New Dest Qty: {holdings[to_key]['qty']:.4f}"
                            )
        except Exception as e_h:
            if IS_DEBUG_DATE:
                logging.error(
                    f"      ERROR processing holding qty for {holding_key_from_row} on row index {index}: {e_h}"
                )
            pass

    cash_summary: Dict[str, Dict] = {}
    # --- Apply STOCK_QUANTITY_CLOSE_TOLERANCE to stock holdings before valuation ---
    for holding_key_iter, data_iter in holdings.items():
        sym_iter, _ = holding_key_iter
        if sym_iter != CASH_SYMBOL_CSV:  # Only for stocks
            qty_iter = data_iter.get("qty", 0.0)
            if 0 < abs(qty_iter) < STOCK_QUANTITY_CLOSE_TOLERANCE:
                if IS_DEBUG_DATE:
                    logging.debug(
                        f"  Applying tolerance to {holding_key_iter}, qty {qty_iter} -> 0"
                    )
                data_iter["qty"] = 0.0
                # Cost basis is not explicitly tracked here for daily valuation, qty is primary
    cash_transactions = transactions_til_date[
        transactions_til_date["Symbol"] == CASH_SYMBOL_CSV
    ].copy()
    if not cash_transactions.empty:

        def get_signed_quantity_cash(row):
            """Calculates cash flow including commission impact."""
            type_lower = str(row.get("Type", "")).lower()
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            commission_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission = 0.0 if pd.isna(commission_raw) else float(commission_raw)
            total_amount_raw = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            total_amount = 0.0 if pd.isna(total_amount_raw) else float(total_amount_raw)

            # If it's a fee or tax transaction
            if type_lower in ["fees", "tax"]:
                fee_val = abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) if pd.notna(qty) and abs(qty) > 1e-9 else abs(commission))
                return -fee_val

            return (
                0.0
                if pd.isna(qty)
                else (
                    # Deposit: Increase cash by quantity MINUS commission
                    abs(qty) - commission
                    if type_lower in ["buy", "deposit", "dividend", "interest"]
                    # Withdrawal: Decrease cash by quantity PLUS commission
                    else (
                        -(abs(qty) + commission)
                        if type_lower in ["sell", "withdrawal"]
                        else 0.0
                    )
                )
            )

        cash_transactions["SignedQuantity"] = cash_transactions.apply(
            get_signed_quantity_cash, axis=1
        )
        cash_qty_agg = cash_transactions.groupby("Account")["SignedQuantity"].sum()
        cash_currency_map = cash_transactions.groupby("Account")[
            "Local Currency"
        ].first()
        all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index)
        for acc in all_cash_accounts:
            cash_summary[acc] = {
                "qty": cash_qty_agg.get(acc, 0.0),
                "local_currency": cash_currency_map.get(acc, default_currency),
                "is_stock": False,
            }

    all_positions: Dict[Tuple[str, str], Dict] = {
        **holdings,
        **{(CASH_SYMBOL_CSV, acc): data for acc, data in cash_summary.items()},
    }
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False
    if IS_DEBUG_DATE:
        logging.debug(
            f"  Value Aggregation Start - Combined Positions ({len(all_positions)}): {list(all_positions.keys())}"
        )

    for (internal_symbol, account), data in all_positions.items():
        current_qty = data.get("qty", 0.0)
        local_currency = data.get("local_currency", default_currency)
        is_stock = data.get("is_stock", internal_symbol != CASH_SYMBOL_CSV)
        DO_DETAILED_LOG = IS_DEBUG_DATE
        if DO_DETAILED_LOG:
            logging.debug(
                f"    Value Agg: Processing {internal_symbol}/{account}, Qty: {current_qty:.4f}"
            )
        if abs(current_qty) < 1e-9:
            continue
            
        # --- ADDED: Filter by included_accounts ---
        # Filter by included_accounts
        if included_accounts and account not in included_accounts_norm:
            continue
        # --- END ADDED ---

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        if DO_DETAILED_LOG:
            logging.debug(
                f"      FX Rate ({local_currency}->{target_currency}): {fx_rate}"
            )
        if pd.isna(fx_rate):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            logging.warning(
                f"Valuation NaN for {internal_symbol}/{account} on {target_date}: "
                f"FX lookup failed ({local_currency}->{target_currency}). "
                f"Whole-day total set to NaN."
            )
            if DO_DETAILED_LOG:
                logging.debug("      CRITICAL: FX lookup failed. Aborting.")
            break

        current_price_local = np.nan
        _manual_price_override_applied = False

        # --- ADDED: Check for manual price override ---
        if manual_overrides_dict and internal_symbol in manual_overrides_dict:
            symbol_override_data = manual_overrides_dict[internal_symbol]
            manual_price = symbol_override_data.get("price")
            if manual_price is not None and pd.notna(manual_price):
                try:
                    manual_price_float = float(manual_price)
                    if manual_price_float > 1e-9:  # Ensure positive price
                        current_price_local = manual_price_float
                        _manual_price_override_applied = True
                        if DO_DETAILED_LOG:
                            logging.debug(
                                f"      Using MANUAL OVERRIDE Price for {internal_symbol}: {current_price_local}"
                            )
                    elif DO_DETAILED_LOG:
                        logging.debug(
                            f"      Manual override price for {internal_symbol} is not positive: {manual_price_float}. Ignoring."
                        )
                except (ValueError, TypeError) as e_manual_price:
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Manual override price for {internal_symbol} ('{manual_price}') is not a valid number: {e_manual_price}. Ignoring."
                        )
        # --- END ADDED ---

        force_fallback = internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
        if (
            pd.isna(current_price_local) and not is_stock
        ):  # If no manual override and it's cash
            current_price_local = 1.0
        elif (
            pd.isna(current_price_local) and not force_fallback and is_stock
        ):  # If no manual override, not forced fallback, and is stock
            yf_symbol_for_lookup = internal_to_yf_map.get(internal_symbol)
            if yf_symbol_for_lookup:
                price_val = get_historical_price(
                    yf_symbol_for_lookup, target_date, historical_prices_yf_unadjusted
                )
                if price_val is not None and pd.notna(price_val) and price_val > 1e-9:
                    current_price_local = float(price_val)
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Using YFinance Price for {internal_symbol}: {current_price_local}"
                        )

        if (pd.isna(current_price_local) or force_fallback) and is_stock:
            # Fallback to last transaction price if still no price, or if yfinance is excluded (and no manual override was applied for it)
            
            # --- ADDED: Check last_known_prices first ---
            if (internal_symbol, account) in last_known_prices:
                last_known = last_known_prices[(internal_symbol, account)]
                if pd.notna(last_known) and last_known > 1e-9:
                    current_price_local = last_known
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Using Last Known Price (tracked): {current_price_local}"
                        )
            # --- END ADDED ---

            if pd.isna(current_price_local):
                try:
                    fallback_tx = transactions_df[
                        (transactions_df["Symbol"] == internal_symbol)
                        & (transactions_df["Account"] == account)
                        & (transactions_df["Price/Share"].notna())
                        & (
                            pd.to_numeric(transactions_df["Price/Share"], errors="coerce")
                            > 1e-9
                        )
                        & (transactions_df["Date"].dt.date <= target_date)
                    ].copy()
                    if not fallback_tx.empty:
                        fallback_tx.sort_values(
                            by=["Date", "original_index"], inplace=True, ascending=True
                        )
                        last_tx_row = fallback_tx.iloc[-1]
                        last_tx_price = pd.to_numeric(
                            last_tx_row["Price/Share"], errors="coerce"
                        )
                        if pd.notna(last_tx_price) and last_tx_price > 1e-9:
                            current_price_local = float(last_tx_price)
                            if DO_DETAILED_LOG:
                                logging.debug(  # This log might be hit if force_fallback=True and no manual price
                                    f"      Using Fallback Price (DF lookup): {current_price_local}"
                                )
                except Exception:
                    pass

        if pd.isna(current_price_local):
            logging.warning(
                f"Missing price for {internal_symbol} on {target_date}. Using 0.0."
            )
            current_price_local = 0.0
        else:
            if DO_DETAILED_LOG:
                logging.debug(f"      Final Local Price: {current_price_local:.4f}")

        market_value_local = current_qty * float(current_price_local)
        market_value_display = market_value_local * fx_rate
        if DO_DETAILED_LOG:
            logging.debug(
                f"      MV Local: {market_value_local:.2f}, MV Display ({target_currency}): {market_value_display:.2f}"
            )
        if pd.isna(market_value_display):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            logging.warning(
                f"Valuation NaN for {internal_symbol}/{account} on {target_date}: "
                f"qty={current_qty:.4f}, price={current_price_local}, fx={fx_rate}. "
                f"Whole-day total set to NaN."
            )
            if DO_DETAILED_LOG:
                logging.debug("      CRITICAL: MV Display is NaN. Aborting.")
            break
        else:
            total_market_value_display_curr_agg += market_value_display
            if DO_DETAILED_LOG:
                logging.debug(
                    f"      Running Total MV Display: {total_market_value_display_curr_agg:.2f}"
                )

    if IS_DEBUG_DATE:
        logging.debug(
            f"--- DEBUG VALUE CALC for {target_date} END --- Final Value: {total_market_value_display_curr_agg}, Lookup Failed: {any_lookup_nan_on_date}"
        )
    return total_market_value_display_curr_agg, any_lookup_nan_on_date


@numba.jit(nopython=True, fastmath=True, cache=True)
def _calculate_holdings_numba(
    target_date_ordinal,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_accounts_np,
    tx_to_accounts_np,  # NEW argument
    tx_types_np,
    tx_quantities_np,
    tx_prices_np,
    tx_totals_np,       # Total Amount per tx; primary fallback for cash math
    tx_commissions_np,
    tx_split_ratios_np,
    tx_local_currencies_np,
    num_symbols,
    num_accounts,
    num_currencies,
    split_type_id,
    stock_split_type_id,
    buy_type_id,
    deposit_type_id,
    sell_type_id,
    withdrawal_type_id,
    short_sell_type_id,
    buy_to_cover_type_id,  # type: ignore
    transfer_type_id,  # NEW argument
    fees_type_id,
    dividend_type_id,
    interest_type_id,
    tax_type_id,        # AUTO CASH
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,     # AUTO CASH: int64 array, 0=Manual, 1=Auto
):
    # Initialize state arrays
    holdings_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_cost_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_currency_np = np.full((num_symbols, num_accounts), -1, dtype=np.int64)

    holdings_short_proceeds_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_short_orig_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    cash_balances_np = np.zeros(num_accounts, dtype=np.float64)
    cash_currency_np = np.full(num_accounts, -1, dtype=np.int64)
    
    # --- NEW: Track last prices ---
    last_prices_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    num_transactions = len(tx_dates_ordinal_np)

    for i in range(num_transactions):
        tx_date = tx_dates_ordinal_np[i]
        if tx_date > target_date_ordinal:
            continue

        symbol_id = tx_symbols_np[i]
        account_id = tx_accounts_np[i]
        type_id = tx_types_np[i]
        qty = tx_quantities_np[i]
        price = tx_prices_np[i]
        total_amount = tx_totals_np[i]
        commission = tx_commissions_np[i]
        split_ratio = tx_split_ratios_np[i]
        currency_id = tx_local_currencies_np[i]

        # --- Handle CASH transactions ---
        if symbol_id == cash_symbol_id:
            if cash_currency_np[account_id] == -1:
                cash_currency_np[account_id] = currency_id

            # Cash-amount fallback chain — see chronological version for the
            # convention rationale. Total Amount preferred; falls back to
            # Quantity (Style A) or commission for fee-style rows.
            cash_amt = abs(total_amount) if abs(total_amount) > 1e-9 else abs(qty)
            if type_id == buy_type_id or type_id == deposit_type_id or type_id == dividend_type_id or type_id == interest_type_id:
                cash_balances_np[account_id] += cash_amt - commission
            elif type_id == sell_type_id or type_id == withdrawal_type_id:
                cash_balances_np[account_id] -= cash_amt + commission
            elif type_id == fees_type_id or type_id == tax_type_id:
                fee_val = abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) if abs(qty) > 1e-9 else abs(commission))
                cash_balances_np[account_id] -= fee_val
            elif type_id == transfer_type_id:
                dest_account_id = tx_to_accounts_np[i]
                if dest_account_id != -1:
                    if cash_currency_np[dest_account_id] == -1:
                        cash_currency_np[dest_account_id] = currency_id

                    cash_balances_np[account_id] -= cash_amt + commission
                    cash_balances_np[dest_account_id] += cash_amt
            continue
        # --- Handle STOCK transactions ---
        if holdings_currency_np[symbol_id, account_id] == -1:
            holdings_currency_np[symbol_id, account_id] = currency_id

        # Update Last Price ONLY for transaction types where Price/Share is the
        # stock price. Dividend / Tax / Interest / Fee rows store the per-share
        # dividend or fee amount in that column, which would poison the
        # last-known-price fallback and create phantom valuation jumps on days
        # where yfinance returns NaN for that symbol.
        if price > 1e-9 and (
            type_id == buy_type_id
            or type_id == sell_type_id
            or type_id == short_sell_type_id
            or type_id == buy_to_cover_type_id
            or type_id == transfer_type_id
        ):
            last_prices_np[symbol_id, account_id] = price

        # --- STOCK TRANSFER LOGIC ---
        if type_id == transfer_type_id:
            if qty > 1e-9:
                source_qty = holdings_qty_np[symbol_id, account_id]
                source_cost_basis = holdings_cost_np[symbol_id, account_id]

                # Determine the quantity to transfer. Do not cap it. Trust the transaction.
                transfer_qty = qty
                cost_to_transfer = 0.0

                # Calculate proportional cost to transfer.
                if abs(source_qty) > 1e-9:  # Avoid division by zero
                    # If transferring more than held (e.g., due to same-day buy), transfer 100% of cost.
                    proportion = min(transfer_qty / source_qty, 1.0)
                    cost_to_transfer = source_cost_basis * proportion

                # 1. Deduct from Source Account
                holdings_qty_np[symbol_id, account_id] -= transfer_qty
                holdings_cost_np[symbol_id, account_id] -= cost_to_transfer

                # Zero out if quantity becomes negligible
                if (
                    abs(holdings_qty_np[symbol_id, account_id])
                    < stock_qty_close_tolerance
                ):
                    holdings_qty_np[symbol_id, account_id] = 0.0
                    holdings_cost_np[symbol_id, account_id] = 0.0

                # 2. Add to Destination
                dest_account_id = tx_to_accounts_np[i]
                if dest_account_id != -1:
                    if holdings_currency_np[symbol_id, dest_account_id] == -1:
                        holdings_currency_np[symbol_id, dest_account_id] = currency_id
                    holdings_qty_np[symbol_id, dest_account_id] += transfer_qty
                    holdings_cost_np[symbol_id, dest_account_id] += cost_to_transfer
                    
                    # Also copy last price to destination if available
                    if last_prices_np[symbol_id, account_id] > 1e-9:
                        last_prices_np[symbol_id, dest_account_id] = last_prices_np[symbol_id, account_id]
            continue

        # --- Existing Split Logic ---
        if type_id == split_type_id or type_id == stock_split_type_id:
            if split_ratio > 1e-9:
                for acc_idx in range(num_accounts):
                    if abs(holdings_qty_np[symbol_id, acc_idx]) > 1e-9:
                        holdings_qty_np[symbol_id, acc_idx] *= split_ratio

                        # Handle Shorts
                        is_shortable = False
                        for short_id in shortable_symbol_ids:
                            if symbol_id == short_id:
                                is_shortable = True
                                break
                        if holdings_qty_np[symbol_id, acc_idx] < -1e-9 and is_shortable:
                            holdings_short_orig_qty_np[
                                symbol_id, acc_idx
                            ] *= split_ratio

            if abs(commission) > 1e-9:
                holdings_cost_np[symbol_id, account_id] += commission
            continue

        # --- Existing Shorting Logic ---
        is_shortable_flag = False
        for short_id in shortable_symbol_ids:
            if symbol_id == short_id:
                is_shortable_flag = True
                break

        if is_shortable_flag and (
            type_id == short_sell_type_id or type_id == buy_to_cover_type_id
        ):
            qty_abs = abs(qty)
            if qty_abs > 1e-9:
                if type_id == short_sell_type_id:
                    proceeds = (qty_abs * price) - commission
                    holdings_qty_np[symbol_id, account_id] -= qty_abs
                    holdings_short_proceeds_np[symbol_id, account_id] += proceeds
                    holdings_short_orig_qty_np[symbol_id, account_id] += qty_abs
                    holdings_cost_np[symbol_id, account_id] += commission
                elif type_id == buy_to_cover_type_id:
                    qty_currently_short = (
                        abs(holdings_qty_np[symbol_id, account_id])
                        if holdings_qty_np[symbol_id, account_id] < -1e-9
                        else 0.0
                    )
                    if qty_currently_short > 1e-9:
                        qty_covered = min(qty_abs, qty_currently_short)
                        cost_to_cover = (qty_covered * price) + commission
                        holdings_qty_np[symbol_id, account_id] += qty_covered
                        holdings_cost_np[symbol_id, account_id] += cost_to_cover

                        short_orig = holdings_short_orig_qty_np[symbol_id, account_id]
                        if short_orig > 1e-9:
                            ratio = qty_covered / short_orig
                            holdings_short_proceeds_np[symbol_id, account_id] *= (
                                1.0 - ratio
                            )
                            holdings_short_orig_qty_np[
                                symbol_id, account_id
                            ] -= qty_covered
            continue

        # --- Standard Buy/Sell ---
        if type_id == buy_type_id or type_id == deposit_type_id:
            if qty > 1e-9:
                cost = (qty * price) + commission
                holdings_qty_np[symbol_id, account_id] += qty
                holdings_cost_np[symbol_id, account_id] += cost
        elif type_id == sell_type_id or type_id == withdrawal_type_id:
            if qty > 1e-9:
                held_qty = holdings_qty_np[symbol_id, account_id]
                if held_qty > 1e-9:
                    qty_sold = min(qty, held_qty)
                    cost_basis_held = holdings_cost_np[symbol_id, account_id]
                    cost_sold = qty_sold * (cost_basis_held / held_qty)

                    holdings_qty_np[symbol_id, account_id] -= qty_sold
                    holdings_cost_np[symbol_id, account_id] -= cost_sold

                    if abs(holdings_qty_np[symbol_id, account_id]) < 1e-9:
                        holdings_qty_np[symbol_id, account_id] = 0.0
                        holdings_cost_np[symbol_id, account_id] = 0.0
        elif type_id == fees_type_id:
            if abs(commission) > 1e-9:
                holdings_cost_np[symbol_id, account_id] += commission

        # --- AUTO CASH LOGIC (Single-date valuation) ---
        # Mirror the conventions used in the chronological version: prefer
        # Total Amount for cash math; fall back to qty * price (or price
        # alone for dividends entered with the legacy "price holds amount"
        # convention).
        if acc_cash_modes[account_id] == 1 and cash_symbol_id >= 0:
            cash_delta = 0.0
            if type_id == buy_type_id:
                cash_delta = -(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price + commission))
            elif type_id == sell_type_id:
                cash_delta = +(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price - commission))
            elif type_id == dividend_type_id:
                div_amt = abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * abs(price) if abs(qty) > 1e-9 and abs(price) > 1e-9 else abs(price))
                cash_delta = +div_amt
            elif type_id == interest_type_id:
                cash_delta = +(abs(total_amount) if abs(total_amount) > 1e-9 else abs(price))
            elif type_id == fees_type_id or type_id == tax_type_id:
                cash_delta = -(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(commission) if abs(commission) > 1e-9 else abs(price)))
            elif type_id == short_sell_type_id:
                cash_delta = +(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price - commission))
            elif type_id == buy_to_cover_type_id:
                cash_delta = -(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price + commission))
            # Deposit, Withdrawal, Split, Transfer: cash_delta stays 0.0

            if abs(cash_delta) > 1e-9:
                cash_balances_np[account_id] += cash_delta

    # Apply Final Tolerance
    for s_id in range(num_symbols):
        if s_id == cash_symbol_id:
            continue
        for a_id in range(num_accounts):
            if 0 < abs(holdings_qty_np[s_id, a_id]) < stock_qty_close_tolerance:
                holdings_qty_np[s_id, a_id] = 0.0
                holdings_cost_np[s_id, a_id] = 0.0

    return (
        holdings_qty_np,
        holdings_cost_np,
        holdings_currency_np,
        cash_balances_np,
        cash_currency_np,
        last_prices_np,  # NEW return
    )


@profile
@numba.jit(nopython=True, fastmath=True, cache=True)
def _calculate_daily_holdings_chronological_numba(
    date_ordinals_np,
    tx_dates_ordinal_np,
    tx_symbols_np,  # type: ignore
    tx_to_accounts_np,  # NEW argument
    tx_accounts_np,
    tx_types_np,
    tx_quantities_np,
    tx_commissions_np,
    tx_split_ratios_np,
    tx_prices_np,  # NEW argument
    tx_totals_np,  # BUG-06 FIX: Total Amount for accurate Auto Cash deltas
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
    transfer_type_id,  # NEW argument
    dividend_type_id,   # AUTO CASH
    interest_type_id,   # AUTO CASH
    fees_type_id,       # AUTO CASH
    tax_type_id,        # AUTO CASH
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,     # AUTO CASH: int64 array, 0=Manual, 1=Auto
):
    """
    Calculates holdings and cash balances chronologically for each day in the target range.
    This is much more efficient than recalculating from scratch each day.
    """
    num_days = len(date_ordinals_np)
    # Initialize daily result arrays
    daily_holdings_qty_np = np.zeros(
        (num_days, num_symbols, num_accounts), dtype=np.float64
    )
    daily_cash_balances_np = np.zeros((num_days, num_accounts), dtype=np.float64)
    # --- NEW: Track last transaction prices for fallback ---
    daily_last_prices_np = np.zeros(
        (num_days, num_symbols, num_accounts), dtype=np.float64
    )

    # Initialize current state
    current_holdings_qty = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    current_cash_balances = np.zeros(num_accounts, dtype=np.float64)
    current_last_prices = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    tx_idx = 0
    num_transactions = len(tx_dates_ordinal_np)

    for day_idx in range(num_days):
        current_date_ordinal = date_ordinals_np[day_idx]

        # Process all transactions for this day
        while (
            tx_idx < num_transactions
            and tx_dates_ordinal_np[tx_idx] <= current_date_ordinal
        ):
            symbol_id = tx_symbols_np[tx_idx]
            account_id = tx_accounts_np[tx_idx]
            type_id = tx_types_np[tx_idx]
            qty = tx_quantities_np[tx_idx]
            commission = tx_commissions_np[tx_idx]
            split_ratio = tx_split_ratios_np[tx_idx]
            # --- NEW: Get price for fallback ---
            price = tx_prices_np[tx_idx]
            # BUG-06 FIX: Get Total Amount for accurate Auto Cash deltas
            total_amount = tx_totals_np[tx_idx]

            if symbol_id == cash_symbol_id:
                # Cash-amount fallback chain: prefer Total Amount, then
                # Quantity, then commission. Manual entries historically
                # store the amount in Quantity (Style A); PDF imports and
                # some manual entries store it in Total Amount (Style B).
                # Reading both keeps the engine resilient to either.
                cash_amt = abs(total_amount) if abs(total_amount) > 1e-9 else abs(qty)
                if type_id == buy_type_id or type_id == deposit_type_id or type_id == dividend_type_id or type_id == interest_type_id:
                    current_cash_balances[account_id] += cash_amt - commission
                elif type_id == sell_type_id or type_id == withdrawal_type_id:
                    current_cash_balances[account_id] -= cash_amt + commission
                elif type_id == fees_type_id or type_id == tax_type_id:
                    # Account-level fee/tax recorded on $CASH symbol (wire fees,
                    # margin interest, withholding, etc.). Debit cash by Total
                    # Amount (preferred), quantity, or commission as fallback.
                    fee_val = abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) if abs(qty) > 1e-9 else abs(commission))
                    current_cash_balances[account_id] -= fee_val
                elif type_id == transfer_type_id:
                    dest_account_id = tx_to_accounts_np[tx_idx]
                    if dest_account_id != -1:
                        # Move Cash: Deduct from Source, Add to Dest
                        # Assuming commission is paid by source
                        current_cash_balances[account_id] -= cash_amt + commission
                        current_cash_balances[dest_account_id] += cash_amt

            else:
                # Update Last Price ONLY for transaction types where Price/Share
                # is the stock price. See the matching note in the target-date
                # function. Prevents dividend/tax/etc rows from poisoning the
                # last-known-price fallback.
                if price > 1e-9 and (
                    type_id == buy_type_id
                    or type_id == sell_type_id
                    or type_id == short_sell_type_id
                    or type_id == buy_to_cover_type_id
                    or type_id == transfer_type_id
                ):
                    current_last_prices[symbol_id, account_id] = price

                if type_id == split_type_id or type_id == stock_split_type_id:
                    if split_ratio > 1e-9:
                        for acc_idx in range(num_accounts):
                            if abs(current_holdings_qty[symbol_id, acc_idx]) > 1e-9:
                                current_holdings_qty[symbol_id, acc_idx] *= split_ratio
                            # --- FIX: Also adjust the fallback price for the split ---
                            # This prevents valuation spikes if market data is missing on the split day.
                            if abs(current_last_prices[symbol_id, acc_idx]) > 1e-9:
                                current_last_prices[symbol_id, acc_idx] /= split_ratio

                elif type_id == buy_type_id or type_id == deposit_type_id:
                    if qty > 1e-9:
                        current_holdings_qty[symbol_id, account_id] += qty

                elif type_id == sell_type_id or type_id == withdrawal_type_id:
                    if qty > 1e-9:
                        held_qty = current_holdings_qty[symbol_id, account_id]
                        qty_sold = min(qty, held_qty) if held_qty > 1e-9 else 0.0
                        current_holdings_qty[symbol_id, account_id] -= qty_sold

                elif type_id == transfer_type_id:
                    if qty > 1e-9:
                        transfer_qty = qty

                        # 1. Deduct from Source Account
                        current_holdings_qty[symbol_id, account_id] -= transfer_qty

                        # 2. Add to Destination Account
                        dest_account_id = tx_to_accounts_np[tx_idx]
                        if dest_account_id != -1:
                            current_holdings_qty[
                                symbol_id, dest_account_id
                            ] += transfer_qty
                            # Also copy last price to destination if available
                            if current_last_prices[symbol_id, account_id] > 1e-9:
                                current_last_prices[symbol_id, dest_account_id] = current_last_prices[symbol_id, account_id]

                        # Note: This function only tracks quantity, not cost basis.
                        # The cost basis transfer is handled in _calculate_holdings_numba.
                else:
                    is_shortable = False
                    for short_id in shortable_symbol_ids:
                        if symbol_id == short_id:
                            is_shortable = True
                            break
                    if is_shortable:
                        if type_id == short_sell_type_id:
                            current_holdings_qty[symbol_id, account_id] -= abs(qty)
                        elif type_id == buy_to_cover_type_id:
                            qty_currently_short = (
                                abs(current_holdings_qty[symbol_id, account_id])
                                if current_holdings_qty[symbol_id, account_id] < -1e-9
                                else 0.0
                            )
                            qty_covered = min(abs(qty), qty_currently_short)
                            current_holdings_qty[symbol_id, account_id] += qty_covered

                # --- AUTO CASH LOGIC (Historical) ---
                # For Auto-mode accounts, generate implicit cash balance changes
                # Skip: Deposit/Withdrawal (in-kind shares), Split, Transfer
                # BUG-02/03 FIX: Aligned with analyzer's three-tier fallback (total → qty*price ± comm)
                if acc_cash_modes[account_id] == 1 and cash_symbol_id >= 0:
                    cash_delta = 0.0
                    if type_id == buy_type_id:
                        cash_delta = -(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price + commission))
                    elif type_id == sell_type_id:
                        cash_delta = +(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * price - commission))
                    elif type_id == dividend_type_id:
                        # Three accepted conventions for Dividend on a stock
                        # symbol: Total=amount (canonical / PDF import), or
                        # Qty=shares & Price=div_per_share (Style A), or
                        # Price=amount with Qty=0 (legacy). Prefer Total, fall
                        # back through qty*price, then bare price.
                        div_amt = abs(total_amount) if abs(total_amount) > 1e-9 else (abs(qty) * abs(price) if abs(qty) > 1e-9 and abs(price) > 1e-9 else abs(price))
                        cash_delta = +div_amt
                    elif type_id == interest_type_id:
                        cash_delta = +(abs(total_amount) if abs(total_amount) > 1e-9 else abs(price))
                    elif type_id == fees_type_id or type_id == tax_type_id:
                        cash_delta = -(abs(total_amount) if abs(total_amount) > 1e-9 else (abs(commission) if abs(commission) > 1e-9 else abs(price)))
                    elif type_id == short_sell_type_id:
                        cash_delta = +(abs(qty) * price - commission)
                    elif type_id == buy_to_cover_type_id:
                        cash_delta = -(abs(qty) * price + commission)
                    # Deposit, Withdrawal, Split, Transfer: cash_delta stays 0.0

                    if abs(cash_delta) > 1e-9:
                        current_cash_balances[account_id] += cash_delta
            
            tx_idx += 1

        for s_id in range(num_symbols):
            if s_id == cash_symbol_id:
                continue
            for a_id in range(num_accounts):
                qty_val = current_holdings_qty[s_id, a_id]
                if 0 < abs(qty_val) < stock_qty_close_tolerance:
                    current_holdings_qty[s_id, a_id] = 0.0

        daily_holdings_qty_np[day_idx] = current_holdings_qty
        daily_cash_balances_np[day_idx] = current_cash_balances
        # --- NEW: Store daily last prices ---
        daily_last_prices_np[day_idx] = current_last_prices

    return daily_holdings_qty_np, daily_cash_balances_np, daily_last_prices_np


@profile
def _calculate_portfolio_value_at_date_unadjusted_numba(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],
    processed_warnings: set,
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    included_accounts: Optional[List[str]] = None,  # ADDED
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
) -> Tuple[float, bool]:
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices (Numba version).
    """
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        return 0.0, False

    # --- FIX: Sort transactions to ensure chronological processing ---
    transactions_til_date.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )
    
    # --- ADDED: Support Intraday comparison ---
    # If target_date is a full datetime (and not just midnight), use exact comparison
    mask = None
    if isinstance(target_date, datetime):
        # Check if it has non-zero time or if we are in intraday mode (context dependent)
        # But safest is: if it's a datetime, use full comparison.
        # However, for legacy daily calls, target_date might be datetime(2023,1,1,0,0) but we want all day?
        # Actually daily logic usually passes date() objects.
        # Intraday logic passes datetime() objects.
        mask = transactions_df["Date"] <= target_date
    else:
        # Fallback for date objects (Legacy Daily)
        mask = transactions_df["Date"].dt.date <= target_date
        
    transactions_til_date = transactions_df[mask].copy()
    if transactions_til_date.empty:
        return 0.0, False
        
    # Re-sort again just in case (though filtered subset should preserve order)
    transactions_til_date.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )
    # --- Prepare NumPy Inputs (Step 4: Update Data Prep) ---
    # --- ADDED: Prepare included_account_ids set ---
    included_account_ids = set()
    if included_accounts:
        # Normalize included_accounts to match account_to_id keys (which are normalized in generate_mappings)
        normalized_included = [acc.strip().upper() for acc in included_accounts]
        for acc in normalized_included:
            if acc in account_to_id:
                included_account_ids.add(account_to_id[acc])
    # --- END ADDED ---

    try:
        target_date_ts = pd.Timestamp(target_date)
        if target_date_ts.tz is None:
            target_date_ts = target_date_ts.tz_localize('UTC')
        target_date_ordinal = target_date_ts.value
        
        # --- FIX: Define tx_types_np earlier for use in debug block ---
        tx_types_series = (
            transactions_til_date["Type"]
            .str.lower()
            .str.strip()
            .map(type_to_id)
            .fillna(-1)
        )
        tx_types_np = tx_types_series.values.astype(np.int64)

        tx_dates_ordinal_np = np.array(pd.to_datetime(transactions_til_date["Date"], utc=True).values.astype('int64'), dtype=np.int64)
        tx_symbols_series = _normalize_series(transactions_til_date["Symbol"]).map(
            symbol_to_id
        )
        tx_symbols_np = tx_symbols_series.fillna(-1).values.astype(np.int64)

        tx_accounts_series = _normalize_series(transactions_til_date["Account"]).map(
            account_to_id
        )
        tx_accounts_np = tx_accounts_series.fillna(-1).values.astype(np.int64)

        # --- ADDED: Map 'To Account' for transfers ---
        if "To Account" in transactions_til_date.columns:
            # Map 'To Account' to IDs, filling NaN with -1
            tx_to_accounts_series = _normalize_series(
                transactions_til_date["To Account"]
            ).map(account_to_id)
        else:
            tx_to_accounts_series = pd.Series(-1, index=transactions_til_date.index)
        tx_to_accounts_np = tx_to_accounts_series.fillna(-1).values.astype(np.int64)
    
        # --- END ADDED ---

        # --- DEBUG BLOCK 2: Check Mapping ---

        # 1. Check ID Map for specific target
        target_acc = "IBKR Acct. 1".upper().strip()
        logging.debug(
            f"Direct ID lookup for '{target_acc}': {account_to_id.get(target_acc, 'NOT FOUND')}"
        )

        # 2. Dump all account keys to check for whitespace issues
        logging.debug("All Account Keys in Map:")
        for k, v in account_to_id.items():
            logging.debug(f"  '{k}' -> {v}")

        # 3. Check Transfer IDs in Arrays
        transfer_id = type_to_id.get("transfer")
        logging.debug(f"Transfer Type ID: {transfer_id}")
        if transfer_id is not None:
            transfer_indices = np.where(tx_types_np == transfer_id)[0]

            if len(transfer_indices) > 0:
                logging.debug(
                    f"Found {len(transfer_indices)} transfers in NumPy arrays."
                )
                for idx in transfer_indices:
                    src_id = tx_accounts_np[idx]
                    dst_id = tx_to_accounts_np[idx]

                    src_name = id_to_account.get(src_id, "UNKNOWN_ID")
                    dst_name = id_to_account.get(dst_id, "UNKNOWN_ID")

                    logging.debug(
                        f"  Tx Index {idx}: From ID {src_id} ('{src_name}') -> To ID {dst_id} ('{dst_name}')"
                    )

                    if dst_id == -1:
                        logging.debug(
                            "  CRITICAL: Destination ID is -1. The Numba engine will IGNORE this transfer."
                        )
            else:
                logging.debug(
                    "CRITICAL: No transfers found in NumPy arrays (tx_types_np)."
                )
        logging.debug("----------------------------\n")
        # --- END ADDED ---

        tx_quantities_np = (
            transactions_til_date["Quantity"].fillna(0.0).values.astype(np.float64)
        )
        tx_prices_np = (
            transactions_til_date["Price/Share"].fillna(0.0).values.astype(np.float64)
        )
        # Total Amount column drives the cash-math fallback chain. Missing
        # column (older fixtures) → zeros, which keeps callers backward-compat.
        if "Total Amount" in transactions_til_date.columns:
            tx_totals_np = (
                transactions_til_date["Total Amount"].fillna(0.0).values.astype(np.float64)
            )
        else:
            tx_totals_np = np.zeros(len(transactions_til_date), dtype=np.float64)
        tx_commissions_series = transactions_til_date["Commission"].fillna(0.0)
        tx_commissions_np = tx_commissions_series.values.astype(np.float64)
        tx_split_ratios_series = transactions_til_date["Split Ratio"].fillna(0.0)
        tx_split_ratios_np = tx_split_ratios_series.values.astype(np.float64)
        tx_local_currencies_series = (
            transactions_til_date["Local Currency"].map(currency_to_id).fillna(-1)
        )
        tx_local_currencies_np = tx_local_currencies_series.values.astype(np.int64)

        split_type_id = type_to_id.get("split", -1)
        stock_split_type_id = type_to_id.get("stock split", -1)
        buy_type_id = type_to_id.get("buy", -1)
        deposit_type_id = type_to_id.get("deposit", -1)
        sell_type_id = type_to_id.get("sell", -1)
        withdrawal_type_id = type_to_id.get("withdrawal", -1)
        short_sell_type_id = type_to_id.get("short sell", -1)
        buy_to_cover_type_id = type_to_id.get("buy to cover", -1)
        transfer_type_id = type_to_id.get("transfer", -1)  # ADDED
        fees_type_id = type_to_id.get("fees", -1)
        dividend_type_id = type_to_id.get("dividend", -1)
        interest_type_id = type_to_id.get("interest", -1)
        tax_type_id = type_to_id.get("tax", -1)  # AUTO CASH
        cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

        # AUTO CASH: Build acc_cash_modes array
        num_accounts = len(account_to_id)
        _acm_val = account_cash_mode_map if account_cash_mode_map else {}
        acc_cash_modes_val = np.zeros(num_accounts, dtype=np.int64)
        for _acc_name_v, _mode_str_v in _acm_val.items():
            if _acc_name_v in account_to_id and _mode_str_v == "Auto":
                acc_cash_modes_val[account_to_id[_acc_name_v]] = 1

        shortable_symbol_ids = np.array(
            [symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id],
            dtype=np.int64,
        )
        num_symbols = len(symbol_to_id)
        num_accounts = len(account_to_id)
        num_currencies = len(currency_to_id)

    except Exception as e_np_prep:
        logging.error(f"Numba Prep Error for {target_date}: {e_np_prep}")
        return np.nan, True

    # --- DEBUG NUMBA INPUTS ---
    
    # --- Call Numba Helper ---
    try:
        (
            holdings_qty_np,
            holdings_cost_np,
            holdings_currency_np,
            cash_balances_np,
            cash_currency_np,
            last_prices_np,  # NEW return
        ) = _calculate_holdings_numba(
            target_date_ordinal,
            tx_dates_ordinal_np,
            tx_symbols_np,
            tx_accounts_np,
            tx_to_accounts_np,  # Pass to Numba function
            tx_types_np,
            tx_quantities_np,
            tx_prices_np,
            tx_totals_np,       # Total Amount for cash-math fallback
            tx_commissions_np,
            tx_split_ratios_np,
            tx_local_currencies_np,
            num_symbols,
            num_accounts,
            num_currencies,
            split_type_id,
            stock_split_type_id,
            buy_type_id,
            deposit_type_id,
            sell_type_id,
            withdrawal_type_id,
            short_sell_type_id,
            buy_to_cover_type_id,
            transfer_type_id,  # Pass to Numba function
            fees_type_id,
            dividend_type_id,
            interest_type_id,
            tax_type_id,
            cash_symbol_id,
            STOCK_QUANTITY_CLOSE_TOLERANCE,
            shortable_symbol_ids,
            acc_cash_modes_val,
        )
        
    except Exception as e_numba_call:
        logging.error(f"Numba Call Error for {target_date}: {e_numba_call}")
        return np.nan, True

    # --- Valuation Loop (using results from Numba) ---
    # [The rest of the valuation loop logic remains identical to the existing file]
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False

    # Iterate through stock holdings
    stock_indices = np.argwhere(np.abs(holdings_qty_np) > STOCK_QUANTITY_CLOSE_TOLERANCE)
    
    for idx_tuple in stock_indices:
        symbol_id = idx_tuple[0]
        account_id = idx_tuple[1]
        
        # --- ADDED: Filter by included_accounts ---
        if included_accounts and account_id not in included_account_ids:
            continue
        # --- END ADDED ---

        current_qty = holdings_qty_np[symbol_id, account_id]
        last_price = last_prices_np[symbol_id, account_id]
        
        internal_symbol = id_to_symbol[symbol_id]
        account = id_to_account[account_id]
        
        # 3. Get Price
        current_price_local = np.nan
        try:
            # Correctly map internal symbol to YF symbol first
            yf_symbol = internal_to_yf_map.get(internal_symbol, internal_symbol)
            current_price_local = get_historical_price(
                yf_symbol,
                target_date,
                historical_prices_yf_unadjusted,
            )
        except Exception:
            pass
            
        # --- NEW: Fallback to last transaction price ---
        if pd.isna(current_price_local):
            if last_price > 1e-9:
                current_price_local = last_price
            else:
                current_price_local = np.nan # Ensure it's a float/NaN, not None

        currency_id = holdings_currency_np[symbol_id, account_id]
        local_currency = id_to_currency.get(currency_id, default_currency)

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        if pd.isna(fx_rate):
            # Hist Fix: Default to 1.0 if missing FX
            fx_rate = 1.0
            # any_lookup_nan_on_date = True
            # total_market_value_display_curr_agg = np.nan
            # break
            
        if current_price_local is None:
            current_price_local = np.nan

        # Calculate local market value
        market_value_local = current_qty * current_price_local
        
        market_value_display = market_value_local * fx_rate

        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Stock {internal_symbol}/{account}, Qty: {current_qty}, Price: {current_price_local}, MV: {market_value_display}")

        if pd.isna(market_value_display):
            if current_qty == 0:
                # Zero qty should not cause NaN
                market_value_display = 0.0
            else:
                 any_lookup_nan_on_date = True
                 # Identify which position failed so the outer "Valuation failed
                 # for date" warning is actionable. Common cause: a Buy/Transfer
                 # row with Price/Share = 0 (e.g. a stock spinoff entered as
                 # zero-cost shares) leaves last_prices_np[symbol] = 0, so when
                 # yfinance also has no historical row for that ticker (delisted
                 # or short-lived spinoff symbol like KRFT 2012-2015) the
                 # fallback chain produces NaN and the position drops out of
                 # the day's total. Naming the symbol points the user straight
                 # at the offending transaction.
                 logging.warning(
                     f"Valuation NaN for {internal_symbol}/{account} on {target_date}: "
                     f"qty={current_qty:.4f}, yfinance price={current_price_local}, "
                     f"last_tx_price={last_price}, fx={fx_rate}. "
                     f"Add a manual price override or enter the acquiring transaction with a non-zero Price/Share."
                 )
        else:
            total_market_value_display_curr_agg += market_value_display

    if IS_DEBUG_DATE:
        logging.debug(f"    Numba Val Agg: Pre-Cash Total: {total_market_value_display_curr_agg}")

    # --- ADDED: Aggregate Cash Balances from Numba ---
    # Note: If any_lookup_nan_on_date is true from stocks, we might still want to sum cash?
    # Original logic breaks early. Keep consistent?
    # If we patched stock FX, we likely won't break early.
    
    cash_indices = np.argwhere(np.abs(cash_balances_np) > 1e-9)
    for acc_id_tuple in cash_indices:
        acc_id = acc_id_tuple[0]
        
        # --- ADDED: Filter by included_accounts ---
        if included_accounts and acc_id not in included_account_ids:
            continue
        # --- END ADDED ---

        account = id_to_account.get(acc_id)
        if account is None:
            continue

        current_qty = cash_balances_np[acc_id]
        currency_id = cash_currency_np[acc_id]
        local_currency = id_to_currency.get(currency_id, default_currency)

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        
        if pd.isna(fx_rate):
             # Hist Fix: Default to 1.0
             fx_rate = 1.0

        cash_val_display = current_qty * fx_rate
        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Cash {account}, Qty: {current_qty}, FX: {fx_rate}, MV: {cash_val_display}")

        total_market_value_display_curr_agg += cash_val_display
        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Running Total after Cash {account}: {total_market_value_display_curr_agg}")

    if IS_DEBUG_DATE:
        logging.debug(
            f"--- DEBUG VALUE CALC for {target_date} END --- Final Value: {total_market_value_display_curr_agg}, Lookup Failed: {any_lookup_nan_on_date}"
        )
    return total_market_value_display_curr_agg, any_lookup_nan_on_date


def _calculate_portfolio_value_at_date_unadjusted(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    processed_warnings: set,
    # --- ADD MAPPINGS and METHOD ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    method: str = HISTORICAL_CALC_METHOD,
    included_accounts: Optional[List[str]] = None,  # ADDED
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
) -> Tuple[float, bool]:
    """
    Dispatcher function to calculate portfolio value using either Python or Numba method.
    """
    if method in ("numba", "numba_chrono"):
        return _calculate_portfolio_value_at_date_unadjusted_numba(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
            symbol_to_id,
            id_to_symbol,
            account_to_id,
            id_to_account,
            type_to_id,
            currency_to_id,
            id_to_currency,
            included_accounts=included_accounts,  # Pass included_accounts
            account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
        )
    elif method == "python":
        return _calculate_portfolio_value_at_date_unadjusted_python(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
            included_accounts=included_accounts,  # Pass included_accounts
        )
    else:
        import traceback
        traceback.print_stack()
        logging.error(
            f"Invalid calculation method specified: {method}. Defaulting to python."
        )
        return _calculate_portfolio_value_at_date_unadjusted_python(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
        )
