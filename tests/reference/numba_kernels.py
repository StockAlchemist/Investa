"""Frozen reference for the Rust kernels in rust/investa_core.

These are the Numba kernels exactly as they ran in production before the port
(portfolio_valuation_kernels.py and portfolio_analyzer.py at commit bd2f0509),
with only their @numba.jit decorators removed. Executed as plain Python they
are the IEEE-exact meaning of that source, which the Rust port reproduces bit
for bit; tests/test_rust_kernels_parity.py holds it to that.

Never edit these to follow a change in Rust. A deliberate behaviour change
lands in both, with the parity test updated in the same commit.

Deliberate changes since the freeze:
- 2026-10-24, _process_numba_core: removed the trailing `elif typ ==
  TYPE_TAX` block. It was the original tax branch (Apr 2026), stranded when
  taxes moved into the Fees/Tax branch (May 2026) and the Auto-cash block was
  inserted above it, which made it an `elif` of that block. Every tax row the
  Auto-cash block skips (Manual-mode accounts, and rows on $CASH) was then
  booked twice: taxes again, dividends reduced by it, historical-FX cost raised.
"""

# ruff: noqa
import numpy as np

TYPE_BUY = 0
TYPE_SELL = 1
TYPE_DEPOSIT = 2
TYPE_WITHDRAWAL = 3
TYPE_DIVIDEND = 4
TYPE_FEES = 5
TYPE_SPLIT = 6
TYPE_TRANSFER = 7
TYPE_SHORT_SELL = 8
TYPE_BUY_TO_COVER = 9
TYPE_TAX = 10
TYPE_INTEREST = 11
TYPE_SPIN_OFF = 12
TYPE_UNKNOWN = -1


def _dividend_amount(qty, price, total_amount):
    """The cash a dividend row pays, read the way the summary reads it.

    Total Amount is the payment whenever it is filled — it is what the broker
    actually paid. Shares x per-share rate only reconstructs it, and on a
    reinvestment the rounded share count makes the product drift, so it is the
    fallback; a bare Price/Share is the legacy form. Mirrors the dividend
    normalization in portfolio_analyzer so the graph and the holdings table
    book the same cash.
    """
    if abs(total_amount) > 1e-9:
        return abs(total_amount)
    if abs(qty) > 1e-9 and abs(price) > 1e-9:
        return abs(qty) * abs(price)
    return abs(price)


def _calculate_holdings_numba(
    target_date_ordinal,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_accounts_np,
    tx_to_accounts_np,  # NEW argument
    tx_types_np,
    tx_quantities_np,
    tx_prices_np,
    tx_totals_np,  # Total Amount per tx; primary fallback for cash math
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
    spin_off_type_id,  # SPIN-OFF: parent basis reduction + child receipt
    fees_type_id,
    dividend_type_id,
    interest_type_id,
    tax_type_id,  # AUTO CASH
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,  # AUTO CASH: int64 array, 0=Manual, 1=Auto
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

            # Column conventions per direction — see the chronological version:
            # money in reads Quantity, money out reads Total Amount.
            cash_in = abs(qty) if abs(qty) > 1e-9 else abs(total_amount)
            cash_out = abs(total_amount) if abs(total_amount) > 1e-9 else abs(qty)
            # Income / fee rows on $CASH move the balance in Auto mode only —
            # see the chronological version for why Manual mode must skip them.
            is_auto_cash = acc_cash_modes[account_id] == 1
            # Commission on a $CASH row is descriptive — see the chronological
            # version. Only the amount moves the balance.
            if type_id == buy_type_id or type_id == deposit_type_id:
                cash_balances_np[account_id] += cash_in
            elif type_id == dividend_type_id:
                if is_auto_cash:
                    cash_balances_np[account_id] += _dividend_amount(
                        qty, price, total_amount
                    )
            elif type_id == interest_type_id:
                if is_auto_cash:
                    cash_balances_np[account_id] += cash_out
            elif type_id == sell_type_id or type_id == withdrawal_type_id:
                cash_balances_np[account_id] -= cash_out
            elif type_id == fees_type_id or type_id == tax_type_id:
                if is_auto_cash:
                    fee_val = (
                        abs(total_amount)
                        if abs(total_amount) > 1e-9
                        else (abs(qty) if abs(qty) > 1e-9 else abs(commission))
                    )
                    cash_balances_np[account_id] -= fee_val
            elif type_id == transfer_type_id:
                dest_account_id = tx_to_accounts_np[i]
                if dest_account_id != -1:
                    if cash_currency_np[dest_account_id] == -1:
                        cash_currency_np[dest_account_id] = currency_id

                    cash_balances_np[account_id] -= cash_out + commission
                    cash_balances_np[dest_account_id] += cash_out
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
            or type_id == spin_off_type_id
        ):
            last_prices_np[symbol_id, account_id] = price

        # --- SPIN-OFF LOGIC ---
        # Cash-neutral corporate action, imported as two legs sharing a date:
        #   - child receipt (qty > 0): create the new position at its allocated
        #     cost basis (Total Amount, or qty * per-share allocated price).
        #   - parent basis reduction (qty == 0): reduce the parent symbol's cost
        #     basis by the amount allocated to the child (Total Amount).
        # See corporate_actions.apply_spin_off for the reference arithmetic.
        if type_id == spin_off_type_id:
            if qty > 1e-9:
                cost = abs(total_amount) if abs(total_amount) > 1e-9 else (qty * price)
                holdings_qty_np[symbol_id, account_id] += qty
                holdings_cost_np[symbol_id, account_id] += cost
            else:
                moved = abs(total_amount)
                current_cost = holdings_cost_np[symbol_id, account_id]
                if moved > current_cost:
                    moved = current_cost
                holdings_cost_np[symbol_id, account_id] = current_cost - moved
            continue

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
                        last_prices_np[symbol_id, dest_account_id] = last_prices_np[
                            symbol_id, account_id
                        ]
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
                            holdings_short_orig_qty_np[symbol_id, acc_idx] *= (
                                split_ratio
                            )
            elif split_ratio < -1e-9:
                # "Target quantity" encoding — see the chronological version.
                # Applies to this account only.
                target_qty = -split_ratio
                old_qty = holdings_qty_np[symbol_id, account_id]
                if abs(old_qty) > 1e-9:
                    implied_ratio = target_qty / old_qty
                    holdings_qty_np[symbol_id, account_id] = target_qty
                    if abs(holdings_short_orig_qty_np[symbol_id, account_id]) > 1e-9:
                        holdings_short_orig_qty_np[symbol_id, account_id] *= (
                            implied_ratio
                        )
                    if (
                        abs(last_prices_np[symbol_id, account_id]) > 1e-9
                        and abs(implied_ratio) > 1e-9
                    ):
                        last_prices_np[symbol_id, account_id] /= implied_ratio

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
                            holdings_short_orig_qty_np[symbol_id, account_id] -= (
                                qty_covered
                            )
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
                cash_delta = -(
                    abs(total_amount)
                    if abs(total_amount) > 1e-9
                    else (abs(qty) * price + commission)
                )
            elif type_id == sell_type_id:
                cash_delta = +(
                    abs(total_amount)
                    if abs(total_amount) > 1e-9
                    else (abs(qty) * price - commission)
                )
            elif type_id == dividend_type_id:
                cash_delta = +_dividend_amount(qty, price, total_amount)
            elif type_id == interest_type_id:
                cash_delta = +(
                    abs(total_amount) if abs(total_amount) > 1e-9 else abs(price)
                )
            elif type_id == fees_type_id or type_id == tax_type_id:
                cash_delta = -(
                    abs(total_amount)
                    if abs(total_amount) > 1e-9
                    else (abs(commission) if abs(commission) > 1e-9 else abs(price))
                )
            elif type_id == short_sell_type_id:
                # See the chronological version: a short on a shortable symbol
                # books its cash through its own $CASH rows, so synthesizing
                # here would count the proceeds twice.
                if not is_shortable_flag:
                    cash_delta = +(
                        abs(total_amount)
                        if abs(total_amount) > 1e-9
                        else (abs(qty) * price - commission)
                    )
            elif type_id == buy_to_cover_type_id:
                if not is_shortable_flag:
                    cash_delta = -(
                        abs(total_amount)
                        if abs(total_amount) > 1e-9
                        else (abs(qty) * price + commission)
                    )
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
    spin_off_type_id,  # SPIN-OFF: child share receipt (qty-only here)
    dividend_type_id,  # AUTO CASH
    interest_type_id,  # AUTO CASH
    fees_type_id,  # AUTO CASH
    tax_type_id,  # AUTO CASH
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
    acc_cash_modes,  # AUTO CASH: int64 array, 0=Manual, 1=Auto
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
                # Which column carries the amount depends on the direction, and
                # these are the summary's conventions (portfolio_analyzer) —
                # the two ledgers have to read a row the same way or the graph
                # drifts away from the headline total.
                #   in  (buy/deposit): Quantity, falling back to Total Amount
                #   out (sell/withdrawal): Total Amount, falling back to Quantity
                # Manual entries historically store the amount in Quantity
                # (Style A); PDF imports store it in Total Amount (Style B), and
                # when both are present they can differ by a fee.
                cash_in = abs(qty) if abs(qty) > 1e-9 else abs(total_amount)
                cash_out = abs(total_amount) if abs(total_amount) > 1e-9 else abs(qty)
                # Income and fee rows booked on $CASH move the balance only in
                # Auto mode. In Manual mode the movement is already recorded by
                # its own $CASH buy/withdrawal row and the dividend/fee row just
                # classifies it, so crediting both counts the money twice.
                is_auto_cash = acc_cash_modes[account_id] == 1
                # The commission on a $CASH row is descriptive, not a second
                # charge: the money it cost leaves through its own $CASH row (a
                # "Commission ..." withdrawal), and the trade leg it mirrors
                # carries the same figure. Only the amount moves the balance —
                # again matching the summary, which books the commission to its
                # own bucket and never touches the cash quantity with it.
                if type_id == buy_type_id or type_id == deposit_type_id:
                    current_cash_balances[account_id] += cash_in
                elif type_id == dividend_type_id:
                    if is_auto_cash:
                        current_cash_balances[account_id] += _dividend_amount(
                            qty, price, total_amount
                        )
                elif type_id == interest_type_id:
                    if is_auto_cash:
                        current_cash_balances[account_id] += cash_out
                elif type_id == sell_type_id or type_id == withdrawal_type_id:
                    current_cash_balances[account_id] -= cash_out
                elif type_id == fees_type_id or type_id == tax_type_id:
                    # Account-level fee/tax recorded on $CASH symbol (wire fees,
                    # margin interest, withholding, etc.). Debit cash by Total
                    # Amount (preferred), quantity, or commission as fallback.
                    if is_auto_cash:
                        fee_val = (
                            abs(total_amount)
                            if abs(total_amount) > 1e-9
                            else (abs(qty) if abs(qty) > 1e-9 else abs(commission))
                        )
                        current_cash_balances[account_id] -= fee_val
                elif type_id == transfer_type_id:
                    dest_account_id = tx_to_accounts_np[tx_idx]
                    if dest_account_id != -1:
                        # Move Cash: Deduct from Source, Add to Dest
                        # Assuming commission is paid by source
                        current_cash_balances[account_id] -= cash_out + commission
                        current_cash_balances[dest_account_id] += cash_out

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
                    or type_id == spin_off_type_id
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
                    elif split_ratio < -1e-9:
                        # Negative ratio is the "target quantity" encoding: the
                        # position *becomes* this many shares, for this account
                        # only (see portfolio_analyzer's split branch, which the
                        # summary uses). Ignoring it strands the difference as a
                        # phantom holding the summary has long since closed.
                        target_qty = -split_ratio
                        old_qty = current_holdings_qty[symbol_id, account_id]
                        if abs(old_qty) > 1e-9:
                            implied_ratio = target_qty / old_qty
                            current_holdings_qty[symbol_id, account_id] = target_qty
                            if (
                                abs(current_last_prices[symbol_id, account_id]) > 1e-9
                                and abs(implied_ratio) > 1e-9
                            ):
                                current_last_prices[symbol_id, account_id] /= (
                                    implied_ratio
                                )

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
                            current_holdings_qty[symbol_id, dest_account_id] += (
                                transfer_qty
                            )
                            # Also copy last price to destination if available
                            if current_last_prices[symbol_id, account_id] > 1e-9:
                                current_last_prices[symbol_id, dest_account_id] = (
                                    current_last_prices[symbol_id, account_id]
                                )

                        # Note: This function only tracks quantity, not cost basis.
                        # The cost basis transfer is handled in _calculate_holdings_numba.

                elif type_id == spin_off_type_id:
                    # Only the child-receipt leg (qty > 0) changes share counts;
                    # the parent basis-reduction leg (qty == 0) is a no-op here.
                    # Cost basis is handled in _calculate_holdings_numba.
                    if qty > 1e-9:
                        current_holdings_qty[symbol_id, account_id] += qty
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

                sym_is_shortable = False
                for short_id in shortable_symbol_ids:
                    if symbol_id == short_id:
                        sym_is_shortable = True
                        break

                # --- AUTO CASH LOGIC (Historical) ---
                # For Auto-mode accounts, generate implicit cash balance changes
                # Skip: Deposit/Withdrawal (in-kind shares), Split, Transfer
                # BUG-02/03 FIX: Aligned with analyzer's three-tier fallback (total → qty*price ± comm)
                if acc_cash_modes[account_id] == 1 and cash_symbol_id >= 0:
                    cash_delta = 0.0
                    if type_id == buy_type_id:
                        cash_delta = -(
                            abs(total_amount)
                            if abs(total_amount) > 1e-9
                            else (abs(qty) * price + commission)
                        )
                    elif type_id == sell_type_id:
                        cash_delta = +(
                            abs(total_amount)
                            if abs(total_amount) > 1e-9
                            else (abs(qty) * price - commission)
                        )
                    elif type_id == dividend_type_id:
                        cash_delta = +_dividend_amount(qty, price, total_amount)
                    elif type_id == interest_type_id:
                        cash_delta = +(
                            abs(total_amount)
                            if abs(total_amount) > 1e-9
                            else abs(price)
                        )
                    elif type_id == fees_type_id or type_id == tax_type_id:
                        cash_delta = -(
                            abs(total_amount)
                            if abs(total_amount) > 1e-9
                            else (
                                abs(commission)
                                if abs(commission) > 1e-9
                                else abs(price)
                            )
                        )
                    elif type_id == short_sell_type_id:
                        # A short on a shortable symbol books its cash through
                        # its own $CASH rows; the summary's shortable branch
                        # returns before synthesizing any, so synthesizing here
                        # would count the proceeds twice.
                        if not sym_is_shortable:
                            cash_delta = +(abs(qty) * price - commission)
                    elif type_id == buy_to_cover_type_id:
                        if not sym_is_shortable:
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


def _process_numba_core(
    sym_ids,
    acc_ids,
    type_ids,
    qtys,
    prices,
    comms,
    totals,
    split_ratios,
    to_acc_ids,
    local_curr_ids,
    fx_rates_hist,
    shortable_sym_ids,
    stock_qty_tol,
    cash_sym_id,
    num_tx,
    num_syms,
    num_accs,
    acc_cash_modes,
):
    # State Array: (num_syms, num_accs, 14)
    # 0: qty
    # 1: total_cost_local
    # 2: realized_gain_local
    # 3: dividends_local
    # 4: commissions_local
    # 5: short_proceeds_local
    # 6: short_original_qty
    # 7: total_cost_invested_local
    # 8: cumulative_investment_local
    # 9: total_buy_cost_local
    # 10: total_cost_display_historical_fx
    # 11: local_curr_id (stored as float)
    # 12: realized_gain_display (accumulated at historical FX)
    # 13: taxes_local (separate from commissions; tax withholding on dividends, etc.)

    state = np.zeros((num_syms, num_accs, 14), dtype=np.float64)
    # Initialize currency ID to -1.0 to detect first use
    state[:, :, 11] = -1.0

    # Transfer costs array: index -> unit_cost (NaN if none)
    transfer_costs = np.full(num_tx, np.nan, dtype=np.float64)

    for i in range(num_tx):
        sym = sym_ids[i]
        acc = acc_ids[i]
        typ = type_ids[i]
        qty = qtys[i]
        price = prices[i]
        comm = comms[i]
        total = totals[i]
        split = split_ratios[i]
        to_acc = to_acc_ids[i]
        curr = local_curr_ids[i]
        fx_rate = fx_rates_hist[i]

        if np.isnan(fx_rate):
            fx_rate = 1.0

        # Access state slice
        current_state = state[sym, acc]

        # Initialize/Check Currency
        if current_state[11] == -1.0:
            current_state[11] = float(curr)
        elif abs(current_state[11] - float(curr)) > 0.1:
            # Currency mismatch - skip logic (simulated)
            # In array approach, we just continue, effectively ignoring it for this transaction
            # But we can't easily skip the *loop* for this transaction if we already started?
            # We just won't update the state.
            pass

        # --- SPLIT ---
        if typ == TYPE_SPLIT:
            # Case 1: Standard Split Ratio (Positive) -> Applies GLOBAL to all accounts
            if split > 1e-9:
                # Iterate all accounts for this symbol
                for a_idx in range(num_accs):
                    h_data = state[sym, a_idx]
                    # Check if initialized (currency != -1)
                    if h_data[11] != -1.0:
                        old_qty = h_data[0]
                        if abs(old_qty) >= 1e-9:
                            h_data[0] *= split
                            # Short logic
                            is_shortable = False
                            # Check if sym is in shortable list
                            # O(N) scan on list? Or we can pass shortable_sym_ids as a boolean array?
                            # Optimization: pass shortable_flags array of size num_syms
                            # For now, list scan is okay if list is small.
                            for s_short in shortable_sym_ids:
                                if s_short == sym:
                                    is_shortable = True
                                    break

                            if is_shortable:
                                if old_qty < -1e-9:
                                    h_data[6] *= split
                                    if abs(h_data[6]) < 1e-9:
                                        h_data[6] = 0.0

                            if abs(h_data[0]) < 1e-9:
                                h_data[0] = 0.0

                # Apply split fee to specific account
                if comm != 0:
                    fee_cost = abs(comm)
                    current_state[4] += fee_cost
                    current_state[7] += fee_cost
                    current_state[8] += fee_cost

            # Case 2: Target Quantity (Negative Encoded) -> Applies LOCAL to specific account only
            elif split < -1e-9:
                target_qty = abs(split)
                # current_state is state[sym, acc]
                if current_state[11] != -1.0:
                    old_qty = current_state[0]
                    # Only apply if we hold something, to calculate implied ratio
                    if abs(old_qty) > 1e-9:
                        # Calculate implied ratio for this specific account
                        ratio = target_qty / old_qty

                        # Apply ratio logic LOCALLY
                        current_state[0] = target_qty

                        # Short logic
                        # (Assume target qty is absolute hold count, logic might be complex for short)
                        # For simplicity, scale short original qty by ratio too
                        if abs(current_state[6]) > 1e-9:
                            current_state[6] *= ratio

                        # Note: We do NOT apply this to other accounts.
                        # This safeguards against "E*TRADE reports 700 shares" blowing up "Sharebuilder" account

                    elif abs(old_qty) < 1e-9 and target_qty > 0:
                        # We held 0, now we hold target_qty?
                        # That's a "Set Quantity" or "Add" event, not really a split.
                        # But strictly for splits usually we hold shares already.
                        # If we had 0, and now have 700, implied ratio is infinite.
                        # Treat as just setting quantity, but cost basis remains 0?
                        # Probably best to leave as 0 or set as is.
                        # If it's a split, usually cost basis propagates.
                        # Let's trust the logic that splits happen on existing holdings.
                        pass

                # Apply split fee
                if comm != 0:
                    fee_cost = abs(comm)
                    current_state[4] += fee_cost
                    current_state[7] += fee_cost
                    current_state[8] += fee_cost

            continue

        # --- TRANSFER ---
        if typ == TYPE_TRANSFER:
            if qty > 1e-9 and to_acc != -1 and to_acc != acc:
                from_qty = current_state[0]
                to_state = state[sym, to_acc]

                if sym == cash_sym_id or from_qty >= qty - 1e-9:
                    to_state = state[sym, to_acc]

                    # Initialize To State if needed
                    if to_state[11] == -1.0:
                        to_state[11] = current_state[11]

                    if True:  # GLD Fix
                        if sym == cash_sym_id:
                            cost_transferred = qty
                            cost_transferred_hist = qty * fx_rate
                            invested_transferred = qty
                            cumulative_transferred = qty
                            buy_cost_transferred = qty
                        else:
                            proportion = 0.0
                            if from_qty > 1e-9:
                                proportion = qty / from_qty

                            cost_transferred = current_state[1] * proportion
                            cost_transferred_hist = current_state[10] * proportion
                            invested_transferred = current_state[7] * proportion
                            cumulative_transferred = current_state[8] * proportion
                            buy_cost_transferred = current_state[9] * proportion

                        current_state[0] -= qty
                        current_state[1] -= cost_transferred
                        current_state[10] -= cost_transferred_hist
                        current_state[7] -= invested_transferred
                        current_state[8] -= cumulative_transferred
                        current_state[9] -= buy_cost_transferred

                        if abs(current_state[0]) < stock_qty_tol:
                            current_state[0] = 0.0
                            current_state[1] = 0.0
                            current_state[10] = 0.0
                            current_state[7] = 0.0
                            current_state[8] = 0.0
                            current_state[9] = 0.0

                        to_state[0] += qty
                        to_state[1] += cost_transferred
                        to_state[10] += cost_transferred_hist
                        to_state[7] += invested_transferred
                        to_state[8] += cumulative_transferred
                        to_state[9] += buy_cost_transferred

                        transfer_costs[i] = cost_transferred / qty
            continue

        # --- SPIN-OFF ---
        # Cash-neutral basis reallocation, imported as two legs sharing a date
        # (see corporate_actions.apply_spin_off). The child receipt (qty > 0)
        # creates the new position at its allocated basis; the parent leg
        # (qty == 0) removes that same basis from the parent. Total cost basis is
        # conserved across the two symbols, mirroring a cross-account transfer.
        if typ == TYPE_SPIN_OFF:
            amt = abs(total)
            if abs(qty) > 1e-9:
                # Child receipt — acquire shares at the allocated cost basis.
                current_state[0] += abs(qty)
                current_state[1] += amt
                current_state[7] += amt
                current_state[8] += amt
                current_state[9] += amt
                current_state[10] += amt * fx_rate
            else:
                # Parent basis reduction — clamp so basis can't go negative, and
                # shrink the historical-FX basis proportionally to stay FX-consistent.
                base = current_state[1]
                red = amt if amt < base else base
                proportion = (red / base) if base > 1e-9 else 0.0
                current_state[1] -= red
                current_state[7] -= red
                current_state[8] -= red
                current_state[9] -= red
                current_state[10] -= current_state[10] * proportion
            continue

        # --- STANDARD TYPES ---
        qty_abs = abs(qty)

        # Short Logic
        # Check shortable
        is_shortable = False
        for s_short in shortable_sym_ids:
            if s_short == sym:
                is_shortable = True
                break

        if is_shortable and (typ == TYPE_SHORT_SELL or typ == TYPE_BUY_TO_COVER):
            if typ == TYPE_SHORT_SELL:
                proceeds = (qty_abs * price) - comm
                current_state[0] -= qty_abs
                current_state[5] += proceeds
                current_state[6] += qty_abs
                current_state[4] += comm
                current_state[8] -= proceeds
                current_state[10] -= proceeds * fx_rate

            elif typ == TYPE_BUY_TO_COVER:
                qty_curr_short = (
                    abs(current_state[0]) if current_state[0] < -1e-9 else 0.0
                )
                qty_covered = min(qty_abs, qty_curr_short)
                cost = (qty_covered * price) + comm

                if current_state[6] > 1e-9:
                    # Calculate historical proceeds (negative cost basis) for the covered portion
                    # BEFORE updating the state
                    if abs(current_state[6]) > 1e-9:
                        avg_proceeds_display_neg = current_state[10] / current_state[6]
                        avg_proceeds = current_state[5] / current_state[6]
                    else:
                        avg_proceeds_display_neg = 0.0
                        avg_proceeds = 0.0

                    proceeds_display_attr = -(
                        avg_proceeds_display_neg * qty_covered
                    )  # This is positive value in display currency

                    # Also calculate local proceeds for gain calc (existing logic)
                    proceeds_attr = qty_covered * avg_proceeds
                    gain = proceeds_attr - cost

                    # Update State
                    current_state[0] += qty_covered
                    current_state[5] -= proceeds_attr
                    current_state[6] -= qty_covered
                    current_state[4] += comm
                    current_state[2] += gain
                    current_state[8] += cost

                    # Update historical cost basis (remove the portion covered)
                    # current_state[10] is negative (proceeds). We want to reduce its magnitude.
                    # proceeds_covered_hist_val = avg_proceeds_display_neg * qty_covered (negative)
                    # current_state[10] -= proceeds_covered_hist_val
                    current_state[10] -= avg_proceeds_display_neg * qty_covered

                    # Realized Gain Display Calculation
                    # Gain = Proceeds (Historical) - Cost (Current)
                    cost_display_cover = cost * fx_rate
                    gain_display = proceeds_display_attr - cost_display_cover

                    current_state[12] += gain_display

                    if abs(current_state[6]) < 1e-9:
                        current_state[5] = 0.0
                        current_state[6] = 0.0
                    if abs(current_state[0]) < 1e-9:
                        current_state[0] = 0.0
            continue

        # Buy / Deposit
        if typ == TYPE_BUY or typ == TYPE_DEPOSIT:
            # Prefer 'total' if provided and non-zero (it's the absolute cash/cost impact)
            # otherwise fallback to qty * price + comm
            cost = abs(total) if abs(total) > 1e-9 else (qty_abs * price) + comm
            current_state[0] += qty_abs
            current_state[1] += cost
            current_state[4] += comm
            current_state[7] += cost
            current_state[8] += cost
            current_state[9] += cost
            current_state[10] += cost * fx_rate

        # Sell / Withdrawal
        elif typ == TYPE_SELL or typ == TYPE_WITHDRAWAL:
            held_qty = current_state[0]
            if sym == cash_sym_id:
                # Cash allows negative balances, process fully without held_qty limits
                # Ensure price is at least 1.0 if not specified (for cost basis)
                eff_price = price if abs(price) > 1e-9 else 1.0
                qty_to_deduct = abs(total) if abs(total) > 1e-9 else qty_abs
                cost_sold = qty_to_deduct * eff_price
                cost_sold_hist = cost_sold * fx_rate

                proceeds = (qty_to_deduct * eff_price) - comm
                gain = 0.0
                gain_display = 0.0

                current_state[0] -= qty_to_deduct
                current_state[1] -= cost_sold
                current_state[10] -= cost_sold_hist
                current_state[4] += comm
                current_state[2] += gain
                current_state[12] += gain_display
                current_state[7] -= cost_sold
                current_state[8] -= proceeds

            elif held_qty > 1e-9:
                qty_sold = min(qty_abs, held_qty)

                cost_sold = 0.0
                cost_sold_hist = 0.0
                if abs(current_state[1]) > 1e-9:
                    cost_sold = qty_sold * (current_state[1] / held_qty)
                    cost_sold_hist = qty_sold * (current_state[10] / held_qty)

                proceeds = (
                    abs(total) if abs(total) > 1e-9 else (qty_sold * price) - comm
                )
                gain = proceeds - cost_sold

                # Realized Gain Display Calculation (Long Sell)
                # Proceeds (Display) = Proceeds_Local * FX_Rate_At_Sell
                # Cost (Display) = Cost_Sold_Hist (already in display currency)
                proceeds_display = proceeds * fx_rate
                gain_display = proceeds_display - cost_sold_hist

                current_state[0] -= qty_sold
                current_state[1] -= cost_sold
                current_state[10] -= (
                    cost_sold_hist  # Update total_cost_display_historical_fx
                )
                current_state[4] += comm
                current_state[2] += gain  # Update realized_gain_local
                current_state[12] += gain_display  # Update realized_gain_display
                current_state[7] -= cost_sold
                current_state[8] -= proceeds

                if abs(current_state[0]) < 1e-9:
                    current_state[0] = 0.0
                    current_state[1] = 0.0
                    current_state[10] = 0.0

        # Dividend
        elif typ == TYPE_DIVIDEND:
            div_amt = price
            div_effect = abs(div_amt)
            if current_state[0] < -1e-9:  # Short
                div_effect = -div_effect

            current_state[3] += div_effect
            current_state[4] += comm
            current_state[10] -= div_effect * fx_rate

            # Explicit Cash Handling:
            # In Auto-mode, we update balance for explicit cash dividends (e.g. awards).
            # In Manual-mode, we skip this to avoid double-counting or breaking legacy bookkeeping.
            if sym == cash_sym_id and acc_cash_modes[acc] == 1:
                current_state[0] += div_effect
                if div_effect > 0:
                    current_state[1] += div_effect
                    current_state[7] += div_effect
                    current_state[8] += div_effect
                    current_state[9] += div_effect
                else:
                    abs_div = abs(div_effect)
                    current_state[1] -= abs_div
                    current_state[7] -= abs_div
                    current_state[8] -= abs_div

        # Fees / Tax
        elif typ == TYPE_FEES or typ == TYPE_TAX:
            cost_val = (
                abs(total)
                if abs(total) > 1e-9
                else (abs(comm) if abs(comm) > 1e-9 else abs(price))
            )
            if typ == TYPE_TAX:
                current_state[13] += cost_val  # taxes_local (separate bucket)
            else:
                current_state[4] += cost_val  # commissions_local
            current_state[7] += cost_val
            current_state[8] += cost_val
            current_state[10] += cost_val * fx_rate

            if sym == cash_sym_id and acc_cash_modes[acc] == 1:
                current_state[0] -= cost_val
                current_state[1] -= cost_val
                current_state[7] -= cost_val
                current_state[8] -= cost_val

        # Interest
        elif typ == TYPE_INTEREST:
            int_amt = abs(total) if abs(total) > 1e-9 else abs(price)
            current_state[3] += int_amt
            current_state[4] += comm
            current_state[10] -= int_amt * fx_rate

            if sym == cash_sym_id and acc_cash_modes[acc] == 1:
                current_state[0] += int_amt
                current_state[1] += int_amt
                current_state[7] += int_amt
                current_state[8] += int_amt
                current_state[9] += int_amt

        # --- AUTO CASH LOGIC ---
        # For Auto-mode accounts, generate implicit $CASH balance changes
        # from non-cash stock transactions.
        # Skip: $CASH symbol (explicit), Deposit/Withdrawal (in-kind shares),
        #        Split (no cash impact), Transfer (handled separately).
        if acc_cash_modes[acc] == 1 and sym != cash_sym_id and cash_sym_id >= 0:
            cash_state = state[cash_sym_id, acc]
            # Initialize cash position if needed
            if cash_state[11] == -1.0:
                cash_state[11] = float(curr)

            cash_delta = 0.0
            if typ == TYPE_BUY:
                cash_delta = -(
                    abs(total) if abs(total) > 1e-9 else (qty_abs * price) + abs(comm)
                )
            elif typ == TYPE_SELL:
                cash_delta = +(
                    abs(total) if abs(total) > 1e-9 else (qty_abs * price) - abs(comm)
                )
            elif typ == TYPE_DIVIDEND:
                cash_delta = +abs(price)
            elif typ == TYPE_INTEREST:
                cash_delta = +(abs(total) if abs(total) > 1e-9 else abs(price))
            elif typ == TYPE_FEES or typ == TYPE_TAX:
                cash_delta = -(
                    abs(total)
                    if abs(total) > 1e-9
                    else (abs(comm) if abs(comm) > 1e-9 else abs(price))
                )
            elif typ == TYPE_SHORT_SELL:
                cash_delta = +((qty_abs * price) - abs(comm))
            elif typ == TYPE_BUY_TO_COVER:
                cash_delta = -((qty_abs * price) + abs(comm))
            # Deposit, Withdrawal, Split, Transfer: cash_delta stays 0.0

            if abs(cash_delta) > 1e-9:
                cash_state[0] += cash_delta  # qty
                if cash_delta > 0:
                    cash_state[1] += cash_delta
                    cash_state[7] += cash_delta
                    cash_state[8] += cash_delta
                    cash_state[9] += cash_delta
                    cash_state[10] += cash_delta * fx_rate
                else:
                    abs_delta = abs(cash_delta)
                    cash_state[1] -= abs_delta
                    cash_state[7] -= abs_delta
                    cash_state[8] -= abs_delta
                    cash_state[10] -= abs_delta * fx_rate


    return state, transfer_costs
