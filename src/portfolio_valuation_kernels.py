"""Holdings/portfolio-value kernels and their Python orchestration.

Split from portfolio_logic.py. The two replay kernels run in Rust
(rust/investa_core, imported as `investa_core`); this module prepares their
NumPy inputs and values the results. Their names still say "numba" because
callers import them by those names.
"""

# ruff: noqa: E402
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set, Tuple

import investa_core
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



def _i64(a):
    """An int64, C-contiguous view (or copy) of `a`, as the Rust kernels take."""
    return np.ascontiguousarray(a, dtype=np.int64)


def _f64(a):
    """A float64, C-contiguous view (or copy) of `a`, as the Rust kernels take."""
    return np.ascontiguousarray(a, dtype=np.float64)


try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


def _dividend_amount(qty, price, total_amount):
    """The cash a dividend row pays, read the way the summary reads it.

    Total Amount is the payment whenever it is filled — it is what the broker
    actually paid. Shares x per-share rate only reconstructs it, and on a
    reinvestment the rounded share count makes the product drift, so it is the
    fallback; a bare Price/Share is the legacy form. Mirrors the dividend
    normalization in portfolio_analyzer so the graph and the holdings table
    book the same cash. Computed in Rust (investa_core) alongside the kernels.
    """
    return investa_core.dividend_amount(
        float(qty), float(price), float(total_amount)
    )


def _normalize_series(series):
    """Normalizes a pandas Series to uppercase and trimmed strings."""
    if series.empty:
        return series
    return series.astype(str).str.upper().str.strip()


def _manual_override_price(
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]], symbol: str
) -> Optional[float]:
    """The user's manually entered price for `symbol`, if it is usable.

    Matched case-insensitively: the engine normalizes symbols, the override file
    keeps whatever case the user typed.
    """
    if not manual_overrides_dict or not symbol:
        return None

    override = manual_overrides_dict.get(symbol)
    if override is None:
        wanted = str(symbol).upper().strip()
        for key, value in manual_overrides_dict.items():
            if str(key).upper().strip() == wanted:
                override = value
                break
    if override is None:
        return None

    raw_price = override.get("price") if isinstance(override, dict) else override
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 1e-9 else None


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
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
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
        logging.debug(
            f"  Transactions count up to date: {len(transactions_df[transactions_df['Date'].dt.date <= target_date])}"
        )

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
        if tx_type in (
            "buy",
            "sell",
            "short sell",
            "buy to cover",
            "transfer",
            "spin off",
            "spin-off",
            "spinoff",
        ):
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
                elif pd.notna(split_ratio) and split_ratio < 0:
                    # "Target quantity" encoding — the position becomes this many
                    # shares, in this account only. See the numba kernels and
                    # portfolio_analyzer, which read it the same way.
                    target_holding = holdings.get(holding_key_from_row)
                    if target_holding is not None:
                        old_qty = target_holding["qty"]
                        if abs(old_qty) >= 1e-9:
                            target_holding["qty"] = abs(float(split_ratio))
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
            elif tx_type in ("spin off", "spin-off", "spinoff"):
                # Child receipt (qty > 0) adds the new position; the parent
                # basis-reduction leg (qty == 0) is a no-op for market value.
                # This function tracks quantity only — cost-basis reallocation
                # lives in _calculate_holdings_numba / the analyzer.
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
                            last_known_prices[to_key] = last_known_prices[
                                holding_key_from_row
                            ]
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
        cash_modes = {
            str(acc).upper().strip(): str(mode).lower()
            for acc, mode in (account_cash_mode_map or {}).items()
        }

        def get_signed_quantity_cash(row):
            """Calculates cash flow including commission impact."""
            type_lower = str(row.get("Type", "")).lower()
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            commission_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission = 0.0 if pd.isna(commission_raw) else float(commission_raw)
            total_amount_raw = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            total_amount = (
                0.0 if pd.isna(total_amount_raw) else abs(float(total_amount_raw))
            )
            # Income / fee rows booked on $CASH move the balance in Auto mode
            # only — in Manual mode the movement has its own $CASH row and
            # counting both books the money twice. Matches the summary.
            is_auto_cash = (
                cash_modes.get(str(row.get("Account", "")).upper().strip()) == "auto"
            )
            if (
                type_lower in ["dividend", "interest", "fees", "tax"]
                and not is_auto_cash
            ):
                return 0.0

            # If it's a fee or tax transaction
            if type_lower in ["fees", "tax"]:
                fee_val = (
                    abs(total_amount)
                    if abs(total_amount) > 1e-9
                    else (
                        abs(qty)
                        if pd.notna(qty) and abs(qty) > 1e-9
                        else abs(commission)
                    )
                )
                return -fee_val

            # The commission column on a $CASH row describes the trade the row
            # mirrors; the charge itself has its own row. Only the amount moves
            # the balance — same as the summary and the numba kernels, which
            # also read money in from Quantity and money out from Total Amount.
            qty_abs = 0.0 if pd.isna(qty) else abs(qty)
            if type_lower in ["buy", "deposit", "dividend", "interest"]:
                return qty_abs if qty_abs > 1e-9 else total_amount
            if type_lower in ["sell", "withdrawal"]:
                return -(total_amount if total_amount > 1e-9 else qty_abs)
            return 0.0

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
                            pd.to_numeric(
                                transactions_df["Price/Share"], errors="coerce"
                            )
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


def _calculate_holdings_numba(
    target_date_ordinal,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_accounts_np,
    tx_to_accounts_np,
    tx_types_np,
    tx_quantities_np,
    tx_prices_np,
    tx_totals_np,
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
    transfer_type_id,
    spin_off_type_id,
    fees_type_id,
    dividend_type_id,
    interest_type_id,
    tax_type_id,
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,
):
    """Holdings, cost basis, currencies, cash and last trade prices as of one date.

    Runs in Rust: investa_core.calculate_holdings, a line-for-line port of the
    Numba kernel that used to live here (frozen in tests/reference/). Returns
    (qty[S, A], cost[S, A], currency[S, A], cash[A], cash_currency[A],
    last_prices[S, A]). The business rules behind each branch are documented in
    rust/investa_core/src/holdings.rs.
    """
    return investa_core.calculate_holdings(
        int(target_date_ordinal),
        _i64(tx_dates_ordinal_np),
        _i64(tx_symbols_np),
        _i64(tx_accounts_np),
        _i64(tx_to_accounts_np),
        _i64(tx_types_np),
        _f64(tx_quantities_np),
        _f64(tx_prices_np),
        _f64(tx_totals_np),
        _f64(tx_commissions_np),
        _f64(tx_split_ratios_np),
        _i64(tx_local_currencies_np),
        int(num_symbols),
        int(num_accounts),
        int(num_currencies),
        int(split_type_id),
        int(stock_split_type_id),
        int(buy_type_id),
        int(deposit_type_id),
        int(sell_type_id),
        int(withdrawal_type_id),
        int(short_sell_type_id),
        int(buy_to_cover_type_id),
        int(transfer_type_id),
        int(spin_off_type_id),
        int(fees_type_id),
        int(dividend_type_id),
        int(interest_type_id),
        int(tax_type_id),
        int(cash_symbol_id),
        float(stock_qty_close_tolerance),
        _i64(shortable_symbol_ids),
        _i64(acc_cash_modes),
    )


def _calculate_daily_holdings_chronological_numba(
    date_ordinals_np,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_to_accounts_np,
    tx_accounts_np,
    tx_types_np,
    tx_quantities_np,
    tx_commissions_np,
    tx_split_ratios_np,
    tx_prices_np,
    tx_totals_np,
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
    transfer_type_id,
    spin_off_type_id,
    dividend_type_id,
    interest_type_id,
    fees_type_id,
    tax_type_id,
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,
):
    """Holdings, cash and last trade prices at the end of every day in a range.

    Runs in Rust: investa_core.calculate_daily_holdings_chronological, a
    line-for-line port of the Numba kernel that used to live here (frozen in
    tests/reference/). Transactions must be sorted by date. Returns
    (daily_qty[D, S, A], daily_cash[D, A], daily_last_prices[D, S, A]). Note the
    argument order: destination accounts before source accounts.
    """
    return investa_core.calculate_daily_holdings_chronological(
        _i64(date_ordinals_np),
        _i64(tx_dates_ordinal_np),
        _i64(tx_symbols_np),
        _i64(tx_to_accounts_np),
        _i64(tx_accounts_np),
        _i64(tx_types_np),
        _f64(tx_quantities_np),
        _f64(tx_commissions_np),
        _f64(tx_split_ratios_np),
        _f64(tx_prices_np),
        _f64(tx_totals_np),
        int(num_symbols),
        int(num_accounts),
        int(split_type_id),
        int(stock_split_type_id),
        int(buy_type_id),
        int(deposit_type_id),
        int(sell_type_id),
        int(withdrawal_type_id),
        int(short_sell_type_id),
        int(buy_to_cover_type_id),
        int(transfer_type_id),
        int(spin_off_type_id),
        int(dividend_type_id),
        int(interest_type_id),
        int(fees_type_id),
        int(tax_type_id),
        int(cash_symbol_id),
        float(stock_qty_close_tolerance),
        _i64(shortable_symbol_ids),
        _i64(acc_cash_modes),
    )


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
            target_date_ts = target_date_ts.tz_localize("UTC")
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

        # Nanoseconds, as target_date_ordinal (Timestamp.value) is: pandas 3
        # parses to microseconds, so the units would otherwise differ.
        tx_dates_ordinal_np = np.array(
            pd.to_datetime(transactions_til_date["Date"], utc=True)
            .dt.as_unit("ns")
            .array.asi8,
            dtype=np.int64,
        )
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
                transactions_til_date["Total Amount"]
                .fillna(0.0)
                .values.astype(np.float64)
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
        spin_off_type_id = type_to_id.get("spin off", -1)  # SPIN-OFF
        fees_type_id = type_to_id.get("fees", -1)
        dividend_type_id = type_to_id.get("dividend", -1)
        interest_type_id = type_to_id.get("interest", -1)
        tax_type_id = type_to_id.get("tax", -1)  # AUTO CASH
        cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

        # AUTO CASH: Build acc_cash_modes array
        num_accounts = len(account_to_id)
        _acm_val = account_cash_mode_map if account_cash_mode_map else {}
        acc_cash_modes_val = np.zeros(num_accounts, dtype=np.int64)
        # Match case-insensitively on both sides. account_to_id arrives
        # uppercased from generate_mappings but keeps the user's case from other
        # callers, and the settings file always keeps what the user typed — a
        # strict comparison drops every account whose name isn't already
        # uppercase into Manual without saying so.
        _acc_ids_by_upper = {
            str(_name).upper().strip(): _idx for _name, _idx in account_to_id.items()
        }
        for _acc_name_v, _mode_str_v in _acm_val.items():
            _idx_v = _acc_ids_by_upper.get(str(_acc_name_v).upper().strip())
            if _idx_v is not None and str(_mode_str_v).lower() == "auto":
                acc_cash_modes_val[_idx_v] = 1

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
            tx_totals_np,  # Total Amount for cash-math fallback
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
            spin_off_type_id,  # SPIN-OFF
            fees_type_id,
            dividend_type_id,
            interest_type_id,
            tax_type_id,
            cash_symbol_id,
            STOCK_QUANTITY_CLOSE_TOLERANCE,
            shortable_symbol_ids,
            acc_cash_modes_val,
        )

    except Exception as e_kernel_call:
        logging.error(f"Holdings kernel error for {target_date}: {e_kernel_call}")
        return np.nan, True

    # --- Valuation Loop (using results from Numba) ---
    # [The rest of the valuation loop logic remains identical to the existing file]
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False

    # Iterate through stock holdings
    stock_indices = np.argwhere(
        np.abs(holdings_qty_np) > STOCK_QUANTITY_CLOSE_TOLERANCE
    )

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

        # --- Fallback chain: manual NAV, then the last transaction price ---
        # Same order as the summary (portfolio_analyzer), so a holding with no
        # price feed is worth the same here as in the holdings table.
        if pd.isna(current_price_local):
            manual_price = _manual_override_price(
                manual_overrides_dict, internal_symbol
            )
            if manual_price is not None:
                current_price_local = manual_price
            elif last_price > 1e-9:
                current_price_local = last_price
            else:
                current_price_local = np.nan  # Ensure it's a float/NaN, not None

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
            logging.debug(
                f"    Numba Val Agg: Stock {internal_symbol}/{account}, Qty: {current_qty}, Price: {current_price_local}, MV: {market_value_display}"
            )

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
        logging.debug(
            f"    Numba Val Agg: Pre-Cash Total: {total_market_value_display_curr_agg}"
        )

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
            logging.debug(
                f"    Numba Val Agg: Cash {account}, Qty: {current_qty}, FX: {fx_rate}, MV: {cash_val_display}"
            )

        total_market_value_display_curr_agg += cash_val_display
        if IS_DEBUG_DATE:
            logging.debug(
                f"    Numba Val Agg: Running Total after Cash {account}: {total_market_value_display_curr_agg}"
            )

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
            account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
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
