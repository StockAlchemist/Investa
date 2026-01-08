
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_analyzer.py
 Purpose:       Contains functions for calculating the current portfolio state,
                holdings, cash balances, and summary metrics from transactions
                and market data. (Refactored from portfolio_logic.py)

 Author:        Google Gemini (Derived from portfolio_logic.py)


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import logging
import re
from datetime import (
    date,
    datetime,
)  # Used in _build_summary_rows, _process_transactions...
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict
import numba
from numba import jit, njit, prange, float64, int64, types
from numba.typed import Dict as NumbaDict

# --- Import Configuration ---
try:
    from config import (
        CASH_SYMBOL_CSV,
        SHORTABLE_SYMBOLS,  # Used by _process_transactions...
        STOCK_QUANTITY_CLOSE_TOLERANCE,  # New tolerance
        _AGGREGATE_CASH_ACCOUNT_NAME_,
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
    _AGGREGATE_CASH_ACCOUNT_NAME_ = "Cash"

# --- Import Utilities ---
try:
    from finutils import (
        is_cash_symbol,
        get_conversion_rate,  # Used by _build_summary_rows
        calculate_irr,  # Used by _build_summary_rows
        get_cash_flows_for_symbol_account,  # Used by _build_summary_rows
        safe_sum,  # Used by _calculate_aggregate_metrics
        get_currency_symbol_from_code,
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
# --- Numba Constants ---
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
TYPE_UNKNOWN = -1

# --- Helper to map string types to ints ---
def get_type_id(t):
    t = str(t).lower().strip()
    if t == 'buy': return TYPE_BUY
    if t == 'sell': return TYPE_SELL
    if t == 'deposit': return TYPE_DEPOSIT
    if t == 'withdrawal': return TYPE_WITHDRAWAL
    if t == 'dividend': return TYPE_DIVIDEND
    if t == 'fees': return TYPE_FEES
    if t in ['split', 'stock split']: return TYPE_SPLIT
    if t == 'transfer': return TYPE_TRANSFER
    if t == 'short sell': return TYPE_SHORT_SELL
    if t == 'buy to cover': return TYPE_BUY_TO_COVER
    return TYPE_UNKNOWN

@jit(nopython=True, cache=True)
def _process_numba_core(
    sym_ids, acc_ids, type_ids, qtys, prices, comms, split_ratios, 
    to_acc_ids, local_curr_ids, fx_rates_hist, 
    shortable_sym_ids, stock_qty_tol,
    num_tx, num_syms, num_accs
):
    # State Array: (num_syms, num_accs, 13)
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
    
    state = np.zeros((num_syms, num_accs, 13), dtype=np.float64)
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
            if split > 0:
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
            continue

        # --- TRANSFER ---
        if typ == TYPE_TRANSFER:
            if qty > 1e-9 and to_acc != -1 and to_acc != acc:
                from_qty = current_state[0]
                
                if from_qty >= qty - 1e-9:
                    to_state = state[sym, to_acc]
                    
                    # Initialize To State if needed
                    if to_state[11] == -1.0:
                        to_state[11] = current_state[11]
                    
                    if abs(to_state[11] - current_state[11]) < 0.1:
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
                current_state[10] -= (proceeds * fx_rate)
                
            elif typ == TYPE_BUY_TO_COVER:
                qty_curr_short = abs(current_state[0]) if current_state[0] < -1e-9 else 0.0
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

                    proceeds_display_attr = - (avg_proceeds_display_neg * qty_covered) # This is positive value in display currency
                    
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
                    current_state[10] -= (avg_proceeds_display_neg * qty_covered)
                    
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
            cost = (qty_abs * price) + comm
            current_state[0] += qty_abs
            current_state[1] += cost
            current_state[4] += comm
            current_state[7] += cost
            current_state[8] += cost
            current_state[9] += cost
            current_state[10] += (cost * fx_rate)

        # Sell / Withdrawal
        elif typ == TYPE_SELL or typ == TYPE_WITHDRAWAL:
            held_qty = current_state[0]
            if held_qty > 1e-9:
                qty_sold = min(qty_abs, held_qty)
                
                cost_sold = 0.0
                cost_sold_hist = 0.0
                if abs(current_state[1]) > 1e-9:
                    cost_sold = qty_sold * (current_state[1] / held_qty)
                    cost_sold_hist = qty_sold * (current_state[10] / held_qty)
                
                proceeds = (qty_sold * price) - comm
                gain = proceeds - cost_sold
                
                # Realized Gain Display Calculation (Long Sell)
                # Proceeds (Display) = Proceeds_Local * FX_Rate_At_Sell
                # Cost (Display) = Cost_Sold_Hist (already in display currency)
                proceeds_display = proceeds * fx_rate
                gain_display = proceeds_display - cost_sold_hist
                
                current_state[0] -= qty_sold
                current_state[1] -= cost_sold
                current_state[10] -= cost_sold_hist # Update total_cost_display_historical_fx
                current_state[4] += comm
                current_state[2] += gain # Update realized_gain_local
                current_state[12] += gain_display # Update realized_gain_display
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
            if current_state[0] < -1e-9: # Short
                div_effect = -div_effect
            
            current_state[3] += div_effect
            current_state[4] += comm
            current_state[10] -= (div_effect * fx_rate)

        # Fees
        elif typ == TYPE_FEES:
            fee_cost = abs(comm)
            current_state[4] += fee_cost
            current_state[7] += fee_cost
            current_state[8] += fee_cost
            current_state[10] += (fee_cost * fx_rate)

    return state, transfer_costs

@profile
def _process_transactions_to_holdings(
    transactions_df: pd.DataFrame,
    default_currency: str,
    shortable_symbols: Set[str],
    historical_fx_lookup: Dict[Tuple[date, str], float],
    display_currency_for_hist_fx: str,
    report_date: date,
) -> Tuple[
    Dict[Tuple[str, str], Dict],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Set[int],
    Dict[int, str],
    Dict[int, str],
    Dict[int, float],
    bool,
]:
    logging.debug("Starting Numba-optimized transaction processing (Array-based)...")
    
    if transactions_df.empty:
        return {}, {}, {}, {}, set(), {}, {}, {}, False

    # Create ID mappings
    all_symbols = transactions_df['Symbol'].astype(str).str.strip().unique()
    all_accounts = set(transactions_df['Account'].astype(str).str.strip().unique())
    if 'To Account' in transactions_df.columns:
        all_accounts.update(transactions_df['To Account'].dropna().astype(str).str.strip().unique())
    all_accounts = list(all_accounts)
    all_currencies = transactions_df['Local Currency'].astype(str).str.strip().str.upper().unique()
    
    sym_map = {s: i for i, s in enumerate(all_symbols)}
    acc_map = {a: i for i, a in enumerate(all_accounts)}
    curr_map = {c: i for i, c in enumerate(all_currencies)}
    
    rev_sym_map = {i: s for s, i in sym_map.items()}
    rev_acc_map = {i: a for a, i in acc_map.items()}
    rev_curr_map = {i: c for c, i in curr_map.items()}
    
    n = len(transactions_df)
    num_syms = len(all_symbols)
    num_accs = len(all_accounts)
    # --- Prepare Data for Numba ---
    df = transactions_df.copy()
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df['Account'] = df['Account'].astype(str).str.strip()
    df['Type'] = df['Type'].astype(str).str.strip().str.lower()
    
    sym_ids = df['Symbol'].map(sym_map).fillna(-1).astype(np.int64).values
    acc_ids = df['Account'].map(acc_map).fillna(-1).astype(np.int64).values
    
    type_map_dict = {
        'buy': TYPE_BUY, 'sell': TYPE_SELL, 'deposit': TYPE_DEPOSIT, 
        'withdrawal': TYPE_WITHDRAWAL, 'dividend': TYPE_DIVIDEND, 'fees': TYPE_FEES,
        'split': TYPE_SPLIT, 'stock split': TYPE_SPLIT, 'transfer': TYPE_TRANSFER,
        'short sell': TYPE_SHORT_SELL, 'buy to cover': TYPE_BUY_TO_COVER
    }
    type_ids = df['Type'].map(type_map_dict).fillna(TYPE_UNKNOWN).astype(np.int64).values
    
    qtys = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0).astype(np.float64).values
    
    raw_prices = pd.to_numeric(df['Price/Share'], errors='coerce').fillna(0.0).astype(np.float64).values
    raw_totals = pd.to_numeric(df['Total Amount'], errors='coerce').fillna(0.0).astype(np.float64).values
    
    is_div = (type_ids == TYPE_DIVIDEND)
    mask_div_total = is_div & (np.abs(raw_totals) > 1e-9)
    raw_prices[mask_div_total] = raw_totals[mask_div_total]
    
    mask_div_calc = is_div & (~mask_div_total)
    raw_prices[mask_div_calc] = raw_prices[mask_div_calc] * np.abs(qtys[mask_div_calc])
    
    prices = raw_prices
    
    comms = pd.to_numeric(df['Commission'], errors='coerce').fillna(0.0).astype(np.float64).values
    split_ratios = pd.to_numeric(df['Split Ratio'], errors='coerce').fillna(0.0).astype(np.float64).values
    
    to_acc_ids = np.full(n, -1, dtype=np.int64)
    if 'To Account' in df.columns:
        df['To Account'] = df['To Account'].astype(str).str.strip()
        to_acc_ids = df['To Account'].map(acc_map).fillna(-1).astype(np.int64).values
    
    df['Local Currency'] = df['Local Currency'].astype(str).str.strip().str.upper()
    invalid_curr = ~df['Local Currency'].isin(curr_map)
    if invalid_curr.any():
        if default_currency not in curr_map:
            curr_map[default_currency] = len(curr_map)
            rev_curr_map[len(curr_map)-1] = default_currency
        df.loc[invalid_curr, 'Local Currency'] = default_currency
    
    local_curr_ids = df['Local Currency'].map(curr_map).astype(np.int64).values
    
    dates = df['Date'].dt.date.values
    currs = df['Local Currency'].values
    fx_rates_hist = np.array([
        historical_fx_lookup.get((d, c), 1.0) 
        for d, c in zip(dates, currs)
    ], dtype=np.float64)
    
    shortable_sym_ids_set = set()
    for s in shortable_symbols:
        if s in sym_map:
            shortable_sym_ids_set.add(sym_map[s])
    
    # Numba List for shortable symbols (Set caused import issues)
    from numba.typed import List as NumbaList
    from numba import int64
    # Initialize with dummy int64 to enforce type, then clear
    nb_shortable = NumbaList([int64(0)])
    nb_shortable.clear()
    for sid in shortable_sym_ids_set:
        nb_shortable.append(sid)
        
    final_state, transfer_costs_nb = _process_numba_core(
        sym_ids, acc_ids, type_ids, qtys, prices, comms, split_ratios,
        to_acc_ids, local_curr_ids, fx_rates_hist,
        nb_shortable, STOCK_QUANTITY_CLOSE_TOLERANCE,
        n, num_syms, num_accs
    )

    # --- Tags Aggregation (Outside Numba) ---
    tags_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    if 'Tags' in df.columns:
        # Vectorized approach to collect tags would be complex, simple iteration is fine for metadata
        # or use groupby
        # Let's iterate over rows where Tags is not empty
        df_tags = df[df['Tags'].notna() & (df['Tags'] != "")]
        for _, row in df_tags.iterrows():
            sym = row['Symbol']
            acc = row['Account']
            t_str = str(row['Tags'])
            # Split by comma if multiple tags
            for t in t_str.split(','):
                cleaned_t = t.strip()
                if cleaned_t:
                    tags_map[(sym, acc)].add(cleaned_t)
            
            # Also handle transfer-in side?
            # If it's a transfer, the tag should technically follow?
            # For now, let's just stick to the account on the transaction record.
    
    holdings = {}
    overall_realized_gains_local = defaultdict(float)
    overall_dividends_local = defaultdict(float)
    overall_commissions_local = defaultdict(float)
    
    # Unpack 3D Array
    # Iterate only over initialized entries
    # We can use np.where to find initialized entries (index 11 != -1)
    # But iterating num_syms * num_accs is fast if they are small.
    # If they are large, np.where is better.
    # Let's use simple loops for now, or np.argwhere.
    
    # Using np.argwhere on the currency channel
    active_indices = np.argwhere(final_state[:, :, 11] != -1.0)
    
    for idx in active_indices:
        s_id, a_id = idx
        val = final_state[s_id, a_id]
        
        sym = rev_sym_map[s_id]
        acc = rev_acc_map[a_id]
        curr = rev_curr_map[int(val[11])]
        
        holdings[(sym, acc)] = {
            "qty": val[0],
            "total_cost_local": val[1],
            "realized_gain_local": val[2],
            "dividends_local": val[3],
            "commissions_local": val[4],
            "local_currency": curr,
            "short_proceeds_local": val[5],
            "short_original_qty": val[6],
            "total_cost_invested_local": val[7],
            "cumulative_investment_local": val[8],
            "total_buy_cost_local": val[9],
            "total_cost_display_historical_fx": val[10],
            "total_cost_display_historical_fx": val[10],
            "realized_gain_display": val[12],
            "tags": sorted(list(tags_map.get((sym, acc), set()))),
        }
        
        overall_realized_gains_local[curr] += val[2]
        overall_dividends_local[curr] += val[3]
        overall_commissions_local[curr] += val[4]
        
    transfer_costs = {}
    orig_indices = df['original_index'].values
    # transfer_costs_nb is array of size n
    # Find indices where it is not nan
    valid_tc_indices = np.where(~np.isnan(transfer_costs_nb))[0]
    for idx in valid_tc_indices:
        if idx < len(orig_indices):
            transfer_costs[orig_indices[idx]] = transfer_costs_nb[idx]
            
    return (
        holdings, 
        dict(overall_realized_gains_local), 
        dict(overall_dividends_local), 
        dict(overall_commissions_local), 
        set(), {}, 
        transfer_costs, 
        False 
    )


# --- REVISED: _build_summary_rows (uses MarketDataProvider) ---
@profile
def _build_summary_rows(
    holdings: Dict[Tuple[str, str], Dict],
    current_stock_data: Dict[
        str, Dict[str, Optional[float]]
    ],  # Data from MarketDataProvider
    current_fx_rates_vs_usd: Dict[str, float],  # Data from MarketDataProvider
    current_fx_prev_close_vs_usd: Dict[str, float], # NEW: For Asset Change calc
    display_currency: str,
    default_currency: str,
    transactions_df: pd.DataFrame,  # Filtered transactions for IRR/fallback
    report_date: date,
    shortable_symbols: Set[str],
    user_excluded_symbols: Set[str],
    user_symbol_map: Dict[str, str],  # New: Accept user symbol map
    manual_prices_dict: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], bool, bool]:
    """
    Builds the detailed list of portfolio summary rows, converting values to the display currency.
    (Implementation remains the same as provided previously - relies only on input data and helpers)
    """
    portfolio_summary_rows: List[Dict[str, Any]] = []
    account_local_currency_map: Dict[str, str] = {}
    has_errors = False
    has_warnings = False

    logging.info(f"Calculating final portfolio summary rows in {display_currency}...")

    # --- NEW: Separate cash from stock holdings ---
    cash_holdings_by_currency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    stock_holdings: Dict[Tuple[str, str], Dict] = {}
    
    for holding_key, data in holdings.items():
        symbol, account = holding_key
        if is_cash_symbol(symbol):
            # Group cash holdings by their local currency for aggregation
            currency = data.get("local_currency", default_currency)
            cash_holdings_by_currency[currency].append({"account": account, **data})
        else:
            stock_holdings[holding_key] = data
    # --- END NEW ---

    for holding_key, data in stock_holdings.items():
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
        realized_gain_display_from_holdings = data.get("realized_gain_display", np.nan)
        tags_list = data.get("tags", [])

        account_local_currency_map[account] = local_currency
        stock_data = current_stock_data.get(symbol, {})

        # --- Price Determination ---
        price_source = "Unknown"
        current_price_local = np.nan
        day_change_local = np.nan
        day_change_pct = np.nan

        # --- MODIFIED: Price Determination Logic ---
        if is_cash_symbol(symbol):
            price_source = "Internal (Cash)"
            current_price_local = 1.0
            day_change_local = 0.0
            day_change_pct = 0.0
        else:
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
                    float(day_change_pct_raw)
                    if pd.notna(day_change_pct_raw)
                    else np.nan
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
        # --- END MODIFIED Price Determination ---

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
        market_value_display = (
            market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        # --- Asset Change (Day Change) with FX ---
        # Value Today = Qty * Price_Today_Local * FX_Today
        # Value Yesterday = Qty * (Price_Today_Local - Day_Change_Local) * FX_Yesterday
        day_change_value_display = np.nan
        
        if pd.notna(current_price_local) and pd.notna(fx_rate) and pd.notna(day_change_local):
             # 1. Get FX Previous Close
             fx_prev = np.nan
             if local_currency == display_currency:
                 fx_prev = 1.0
             else:
                 fx_prev = get_conversion_rate(
                    local_currency, display_currency, current_fx_prev_close_vs_usd
                 )
             
             if pd.notna(fx_prev):
                 val_today = current_qty * current_price_local * fx_rate
                 price_yesterday_local = current_price_local - day_change_local
                 val_yesterday = current_qty * price_yesterday_local * fx_prev
                 val_yesterday = current_qty * price_yesterday_local * fx_prev
                 day_change_value_display = val_today - val_yesterday
             else:
                 # Fallback if FX prev missing: Ignore FX change
                 day_change_value_display = current_qty * day_change_local * fx_rate
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
            realized_gain_display_from_holdings 
            if pd.notna(realized_gain_display_from_holdings) 
            else (realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan)
        )
        realized_gain_display = (
            realized_gain_display_from_holdings 
            if pd.notna(realized_gain_display_from_holdings) 
            else (realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan)
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
                transactions_df,  # The filtered transactions DataFrame
                market_value_local_for_irr,  # The final value of this specific holding
                is_transfer_a_flow=True,  # ADDED: Treat transfers as flows for IRR
                report_date=report_date,  # The date of the final value
            )
            if cf_dates and cf_values:
                # Ensure all dates are datetime.date objects to prevent TypeErrors
                # caused by string vs date comparison or calculate_irr validation.
                for i in range(len(cf_dates)):
                    d = cf_dates[i]
                    if isinstance(d, str):
                        try:
                            cf_dates[i] = datetime.strptime(d, "%Y-%m-%d").date()
                        except ValueError:
                            # Try ISO format or just keep as is if parse fails (will likely fail later)
                             pass
                    elif isinstance(d, datetime):
                        cf_dates[i] = d.date()
                
                # --- NEW: Suppress Annualized IRR for short duration (< 1 year) ---
                # Check actual holding duration including transfers (via Lots)
                earliest_lot_date = None
                if "lots" in data and data["lots"]:
                    # Sort lots just in case, though usually sorted
                    sorted_lots = sorted(data["lots"], key=lambda x: x["Date"])
                    earliest_lot_date_str = sorted_lots[0]["Date"]
                    try:
                        earliest_lot_date = datetime.strptime(str(earliest_lot_date_str), "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        # Fallback or log if date parsing fails
                        earliest_lot_date = None

                # Determine effective start date (use lot date if older than first flow date)
                effective_start_date = cf_dates[0]
                if earliest_lot_date and earliest_lot_date < effective_start_date:
                    # If lot date is older, it implies we have history before this account (e.g. transfer)
                    # We shift the start date of the calculation to the lot date to capture true duration.
                    effective_start_date = earliest_lot_date
                    # Modify the cash flow stream: Move the initial flow (Transfer In) to the Lot Date
                    # This approximates the original buy.
                    # Note: We must ensure dates remain sorted. Since earliest_lot_date < cf_dates[0],
                    # and cf_dates is sorted, replacing index 0 keeps it sorted.
                    cf_dates[0] = earliest_lot_date

                duration_days = (cf_dates[-1] - cf_dates[0]).days
                
                if duration_days >= 365:
                    stock_irr = calculate_irr(cf_dates, cf_values)
                else:
                    # Still suppress if true duration is < 1 year
                    # logging.debug(f"IRR suppressed for {symbol}/{account}: Duration {duration_days} days < 365 days.")
                    stock_irr = np.nan
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
            not is_cash_symbol(symbol)  # Not cash
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
                f"Div. Yield (Current) %": div_yield_on_current_pct_display,
                f"Est. Ann. Income ({display_currency})": est_annual_income_display,
                f"FX Gain/Loss ({display_currency})": fx_gain_loss_display_holding,
                "FX Gain/Loss %": fx_gain_loss_pct_holding,
                "Name": stock_data.get("name", ""),  # Add Company Name
                "sparkline_7d": stock_data.get("sparkline_7d", []),
                "Tags": tags_list,  # Added Tags
            }
        )
    # --- End Stock/ETF Loop ---

    # --- Loop 2: Process and Aggregate Cash Holdings ---
    for currency, holding_list in cash_holdings_by_currency.items():
        if not holding_list:
            continue

        # Aggregate values from all accounts for this cash currency
        agg_qty = sum(h.get("qty", 0.0) for h in holding_list)

        # --- ADDED: Skip cash rows with zero quantity ---
        if abs(agg_qty) < STOCK_QUANTITY_CLOSE_TOLERANCE:
            logging.debug(
                f"Skipping cash summary for currency '{currency}' as aggregated quantity is near zero ({agg_qty:.8f})."
            )
            continue
        # --- END ADDED ---

        agg_realized_gain_local = sum(
            h.get("realized_gain_local", 0.0) for h in holding_list
        )
        agg_dividends_local = sum(h.get("dividends_local", 0.0) for h in holding_list)
        agg_commissions_local = sum(
            h.get("commissions_local", 0.0) for h in holding_list
        )
        agg_total_cost_local = sum(h.get("total_cost_local", 0.0) for h in holding_list)
        agg_total_cost_invested_local = sum(
            h.get("total_cost_invested_local", 0.0) for h in holding_list
        )
        agg_cumulative_investment_local = sum(
            h.get("cumulative_investment_local", 0.0) for h in holding_list
        )
        agg_total_buy_cost_local = sum(
            h.get("total_buy_cost_local", 0.0) for h in holding_list
        )
        agg_total_cost_display_historical_fx = np.nansum(
            [h.get("total_cost_display_historical_fx", np.nan) for h in holding_list]
        )

        # For cash, local currency is the currency we grouped by
        local_currency = currency
        account = (
            _AGGREGATE_CASH_ACCOUNT_NAME_  # Use the special aggregate account name
        )

        # --- NEW: Format cash symbol ---
        currency_symbol_for_display = get_currency_symbol_from_code(local_currency)
        cash_display_symbol = f"Cash ({currency_symbol_for_display})"
        # --- END NEW ---

        # --- Now, apply the same logic as for stocks, but with aggregated values ---
        current_qty = agg_qty
        realized_gain_local = agg_realized_gain_local
        dividends_local = agg_dividends_local
        commissions_local = agg_commissions_local
        current_total_cost_local = agg_total_cost_local
        total_cost_invested_local = agg_total_cost_invested_local
        cumulative_investment_local = agg_cumulative_investment_local
        total_buy_cost_local = agg_total_buy_cost_local
        total_cost_display_historical_fx_val = agg_total_cost_display_historical_fx

        # For cash, price is always 1.0 in its local currency
        price_source = "Internal (Cash)"
        current_price_local = 1.0
        day_change_local = 0.0
        day_change_pct = 0.0

        # --- Currency Conversion ---
        fx_rate = get_conversion_rate(
            local_currency, display_currency, current_fx_rates_vs_usd
        )
        if pd.isna(fx_rate):
            logging.error(
                f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for aggregated cash."
            )
            has_errors = True
            fx_rate = np.nan

        # --- Dividend Yield and Income Calculations (Not applicable for cash) ---
        div_yield_on_cost_pct_display = np.nan
        div_yield_on_current_pct_display = np.nan
        est_annual_income_display = np.nan

        # --- Calculate Display Currency Values ---
        market_value_local = current_qty * current_price_local
        market_value_display = (
            market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        # --- Asset Change (Day Change) with FX for Cash ---
        # Value Today = Qty * 1.0 * FX_Today
        # Value Yesterday = Qty * 1.0 * FX_Yesterday (Prev Close)
        day_change_value_display = 0.0
        day_change_pct = 0.0
        
        if local_currency != display_currency and pd.notna(fx_rate) and pd.notna(current_qty):
             # 1. Get FX Previous Close
             fx_prev = np.nan
             # Safe fallback: if local matches display (e.g. USD cash displayed in USD), fx_prev is 1.0
             # But the 'if' condition above handles that.
             
             fx_prev = get_conversion_rate(
                local_currency, display_currency, current_fx_prev_close_vs_usd
             )
             
             if pd.notna(fx_prev):
                 val_today = current_qty * 1.0 * fx_rate
                 val_yesterday = current_qty * 1.0 * fx_prev
                 day_change_value_display = val_today - val_yesterday
                 
                 if abs(val_yesterday) > 1e-9:
                     day_change_pct = (day_change_value_display / val_yesterday) * 100.0
        
        current_price_display = (
            current_price_local * fx_rate
            if pd.notna(current_price_local) and pd.notna(fx_rate)
            else np.nan
        )
        cost_basis_display = (
            market_value_display  # For cash, cost basis is market value
        )
        avg_cost_price_display = (
            current_price_display  # For cash, avg cost is current price
        )
        unrealized_gain_display = 0.0
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

        total_gain_display = (
            (
                realized_gain_display
                + unrealized_gain_display
                + dividends_display
                - commissions_display
            )
            if all(
                pd.notna(v)
                for v in [realized_gain_display, dividends_display, commissions_display]
            )
            else np.nan
        )

        # Total Return % for cash is not meaningful in the same way as for stocks
        total_return_pct = np.nan

        # IRR for cash is not applicable
        stock_irr = np.nan

        # FX Gain/Loss for cash
        fx_gain_loss_display_holding = np.nan
        fx_gain_loss_pct_holding = np.nan
        if local_currency == display_currency:
            fx_gain_loss_display_holding = 0.0
            fx_gain_loss_pct_holding = 0.0
        elif (
            pd.notna(cost_basis_display)
            and abs(cost_basis_display) > 1e-9
            and pd.notna(total_cost_display_historical_fx_val)
        ):
            try:
                fx_gain_loss_display_holding = (
                    cost_basis_display - total_cost_display_historical_fx_val
                )
                if abs(total_cost_display_historical_fx_val) > 1e-9:
                    fx_gain_loss_pct_holding = (
                        fx_gain_loss_display_holding
                        / total_cost_display_historical_fx_val
                    ) * 100.0
                elif abs(fx_gain_loss_display_holding) < 1e-9:
                    fx_gain_loss_pct_holding = 0.0
            except Exception as e_fx_gl_cash:
                logging.error(
                    f"Error calculating FX G/L for cash symbol {CASH_SYMBOL_CSV} ({local_currency}): {e_fx_gl_cash}"
                )

        portfolio_summary_rows.append(
            {
                "Account": account,
                "Symbol": cash_display_symbol,
                "Quantity": current_qty,
                f"Avg Cost ({display_currency})": avg_cost_price_display,
                f"Price ({display_currency})": current_price_display,
                f"Cost Basis ({display_currency})": cost_basis_display,
                f"Market Value ({display_currency})": market_value_display,
                f"Day Change ({display_currency})": day_change_value_display,
                "Day Change %": day_change_pct,
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
                "IRR (%)": stock_irr,
                "Local Currency": local_currency,
                "Price Source": price_source,
                f"Div. Yield (Cost) %": div_yield_on_cost_pct_display,
                f"Div. Yield (Current) %": div_yield_on_current_pct_display,
                f"Est. Ann. Income ({display_currency})": est_annual_income_display,
                f"FX Gain/Loss ({display_currency})": fx_gain_loss_display_holding,
                "FX Gain/Loss %": fx_gain_loss_pct_holding,
                "Name": "Cash",  # Add Name for Cash
            }
        )
    # --- End Cash Loop ---



    # --- NEW: Calculate % of Total for all rows ---
    total_mv_display = sum(
        row.get(f"Market Value ({display_currency})", 0.0) 
        for row in portfolio_summary_rows 
        if pd.notna(row.get(f"Market Value ({display_currency})"))
    )
    
    for row in portfolio_summary_rows:
        mv = row.get(f"Market Value ({display_currency})", 0.0)
        if pd.notna(mv) and abs(total_mv_display) > 1e-9:
            row["pct_of_total"] = (mv / total_mv_display) * 100.0
        else:
            row["pct_of_total"] = 0.0
    # --- END NEW ---

    return (
        portfolio_summary_rows,
        dict(account_local_currency_map),
        has_errors,
        has_warnings,
    )


# --- REVISED: _calculate_aggregate_metrics (Removed transactions_df parameter) ---
# @profile
def _calculate_aggregate_metrics(
    full_summary_df: pd.DataFrame,
    display_currency: str,
    report_date: date,
    include_accounts: Optional[List[str]] = None,  # <-- New param
    all_available_accounts: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Calculates account-level and overall portfolio summary metrics.
    (Implementation remains the same as provided previously - relies only on input df and helpers)
    """
    # Removed debug logging
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
            ]
            for col in cols_to_sum_display:
                if col not in account_full_df.columns:
                    logging.warning(
                        f"Warn: Col '{col}' missing for acc '{account}' agg."
                    )
                    has_warnings = True

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

            # FIX: EXCLUDE FX Gain/Loss from Market Value (It is already intrinsic in Price * FX conversion)
            metrics_entry["total_market_value_display"] = safe_sum(
                account_full_df, f"Market Value ({display_currency})"
            )
            
            metrics_entry["total_realized_gain_display"] = safe_sum(
                account_full_df, f"Realized Gain ({display_currency})"
            )
            # FIX: EXCLUDE FX Gain/Loss from Unrealized Gain (Intrinsic)
            metrics_entry["total_unrealized_gain_display"] = safe_sum(
                account_full_df, f"Unreal. Gain ({display_currency})"
            )

            metrics_entry["total_dividends_display"] = safe_sum(
                account_full_df, f"Dividends ({display_currency})"
            )
            metrics_entry["total_commissions_display"] = safe_sum(
                account_full_df, f"Commissions ({display_currency})"
            )
            # FIX: EXCLUDE FX Gain/Loss from Total Gain (Intrinsic)
            metrics_entry["total_gain_display"] = safe_sum(
                account_full_df, f"Total Gain ({display_currency})"
            )
            
            # Cash is now part of the main holdings, so total_cash_display is not needed here.
            # It can be derived from the main market value if needed, but it's not a primary metric for an account.
            metrics_entry["total_cash_display"] = 0.0
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
            # --- DEBUG LOGGING ---
            # Log the top contributors to Day Change to trace fluctuations
            if abs(acc_total_day_change_display) > 0:
                day_change_breakdown = account_full_df[[
                    "Symbol", f"Day Change ({display_currency})", f"Market Value ({display_currency})"
                ]].copy()
                day_change_breakdown = day_change_breakdown.sort_values(
                    by=f"Day Change ({display_currency})", key=abs, ascending=False
                ).head(5) # Top 5 movers
                
                logging.info(f"Day Change Breakdown for {account}: Total={acc_total_day_change_display}")
                for _, row in day_change_breakdown.iterrows():
                    logging.info(
                        f"  > {row['Symbol']}: DayChg={row[f'Day Change ({display_currency})']}, "
                        f"MV={row[f'Market Value ({display_currency})']}"
                    )
            # ---------------------
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

        except Exception as e_acc_agg:
            logging.exception(f"Error aggregating metrics for account '{account}'")
            has_warnings = True

    # --- START FIX: Filter for Overall Summary ---
    df_for_overall_summary = full_summary_df

    # Determine if the filter is for "All Accounts"
    is_all_accounts_selected = False
    if not include_accounts:  # None or empty list
        is_all_accounts_selected = True
    elif all_available_accounts and set(include_accounts) == set(
        all_available_accounts
    ):
        is_all_accounts_selected = True

    if not is_all_accounts_selected:
        # Filter the DataFrame to *only* the accounts the user selected
        logging.debug(
            f"Aggregating overall metrics for selected accounts: {include_accounts}"
        )
        if "Account" in df_for_overall_summary.columns:
            # Keep rows that match the selected accounts
            account_mask = full_summary_df["Account"].isin(include_accounts)

            # --- MODIFIED: Also keep the special aggregate cash row ---
            # When filtering for specific accounts, we must still include the aggregated cash
            # row to ensure the 'Overall' portfolio metrics (like total market value) are correct.
            cash_mask = full_summary_df["Account"] == _AGGREGATE_CASH_ACCOUNT_NAME_

            # Combine the masks to keep selected accounts AND the cash row.
            df_for_overall_summary = full_summary_df[account_mask | cash_mask]
            # --- END MODIFICATION ---
        else:
            logging.warning("Cannot filter overall metrics: 'Account' column missing.")
    # --- END FIX ---

    # Note: The original code had a second part to the filter for the "All Accounts" case.
    # That logic is now correctly handled by the `show_closed_positions` filter in the main summary function,
    # which removes zero-quantity transfer source rows before they even get here.
    # Therefore, only the explicit account filter is needed for the overall summary calculation.

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
        if col not in df_for_overall_summary.columns:
            logging.warning(f"Warning: Column '{col}' missing for overall aggregation.")
            has_warnings = True
    
    # --- ADDED: Overall FX Gain/Loss ---
    overall_fx_gain_loss_display = safe_sum(df_for_overall_summary, fx_gain_loss_col)
    # --- END ADDED ---

    # FIX: EXCLUDE FX Gain/Loss from Overall Market Value (Intrinsic)
    raw_sum_mkt = safe_sum(df_for_overall_summary, mkt_val_col)
    overall_market_value_display = raw_sum_mkt
    
    held_mask = pd.Series(False, index=df_for_overall_summary.index)
    if (
        "Quantity" in df_for_overall_summary.columns
    ):  # Symbol check is implicit as cash is now a holding
        held_mask = df_for_overall_summary["Quantity"].abs() > 1e-9
    overall_cost_basis_display = (
        safe_sum(df_for_overall_summary.loc[held_mask], cost_basis_col)
        if held_mask.any()
        else 0.0
    )
    # FIX: EXCLUDE FX Gain/Loss from Overall Unrealized Gain (Intrinsic)
    overall_unrealized_gain_display = safe_sum(df_for_overall_summary, unreal_gain_col)
    
    overall_realized_gain_display_agg = safe_sum(df_for_overall_summary, real_gain_col)
    overall_dividends_display_agg = safe_sum(df_for_overall_summary, divs_col)
    overall_commissions_display_agg = safe_sum(df_for_overall_summary, comm_col)
    
    # FIX: EXCLUDE FX Gain/Loss from Overall Total Gain (Intrinsic)
    overall_total_gain_display = safe_sum(df_for_overall_summary, total_gain_col)
    
    overall_total_cost_invested_display = safe_sum(
        df_for_overall_summary, cost_invest_col
    )
    overall_cumulative_investment_display = safe_sum(
        df_for_overall_summary, cum_invest_col
    )
    overall_total_buy_cost_display = safe_sum(
        df_for_overall_summary, total_buy_cost_col
    )
    overall_day_change_display = safe_sum(df_for_overall_summary, day_change_col)
    overall_prev_close_mv_display = np.nan


    # --- ADDED: Overall Estimated Annual Income ---
    est_ann_income_col = f"Est. Ann. Income ({display_currency})"
    overall_est_annual_income_display = safe_sum(
        df_for_overall_summary, est_ann_income_col
    )
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
    overall_denominator = overall_cost_basis_display  # Use cost basis of held assets + cash as the denominator
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
    intervals = {"W": "W-FRI", "M": "ME", "Y": "YE", "D": "D"}

    for interval_key, freq_code in intervals.items():
        try:
            # --- MODIFIED: Include daily_gain for value change calculation ---
            cols_to_resample = list(valid_gain_cols)
            agg_dict = {col: "last" for col in valid_gain_cols}
            
            if "daily_gain" in historical_df.columns:
                cols_to_resample.append("daily_gain")
                agg_dict["daily_gain"] = "sum"
            
            # Resample
            resampled_data = (
                historical_df[cols_to_resample].resample(freq_code).agg(agg_dict)
            )
            
            # Separate factors and value changes
            resampled_factors = resampled_data[valid_gain_cols]
            resampled_value_change = resampled_data["daily_gain"] if "daily_gain" in resampled_data.columns else pd.Series()
            # --- END MODIFIED ---

            # --- ADDED: Filter out weekends for Daily returns ---
            if interval_key == "D":
                # 0=Monday, 6=Sunday. Filter out 5 (Sat) and 6 (Sun)
                # Filter both factors and value change
                mask = resampled_factors.index.dayofweek < 5
                resampled_factors = resampled_factors[mask]
                if not resampled_value_change.empty:
                    resampled_value_change = resampled_value_change[mask]
                
                # Filter out US Holidays
                try:
                    cal = USFederalHolidayCalendar()
                    holidays = cal.holidays(start=resampled_factors.index.min(), end=resampled_factors.index.max())
                    mask_hol = ~resampled_factors.index.isin(holidays)
                    resampled_factors = resampled_factors[mask_hol]
                    if not resampled_value_change.empty:
                        resampled_value_change = resampled_value_change[mask_hol]
                except Exception as e_hol:
                    logging.warning(f"Failed to filter holidays: {e_hol}")
            # --- END ADDED ---

            # --- MODIFIED: Handle first period return correctly ---
            # pct_change() will make the first period NaN, which gets dropped.
            # We need to calculate it manually: (end_factor - 1)
            if not resampled_factors.empty:
                # Calculate returns using pct_change for all periods after the first
                period_returns_df = resampled_factors.pct_change(fill_method=None)

                # Manually calculate the return for the first period
                first_period_return = resampled_factors.iloc[0] - 1.0
                period_returns_df.iloc[0] = first_period_return

                # Convert to percentage
                period_returns_df *= 100.0
                
                # Add Value Change column
                if not resampled_value_change.empty:
                    # Align indices just in case
                    period_returns_df[f"Portfolio {interval_key}-Value"] = resampled_value_change
            else:
                period_returns_df = pd.DataFrame(columns=resampled_factors.columns)
            # --- END MODIFIED ---
            # --- END MODIFIED ---

            # Rename columns for clarity (optional)
            period_returns_df.columns = [
                col.replace(" Accumulated Gain", f" {interval_key}-Return")
                for col in period_returns_df.columns
            ]

            periodic_returns[interval_key] = period_returns_df

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

        if pd.isna(qty) or qty <= 1e-9:
            continue

        fx_rate_to_display_current_tx = get_historical_rate_via_usd_bridge(
            local_curr, display_currency, tx_date, historical_fx_yf
        )

        if tx_type == "buy":
            if pd.isna(price_local) or price_local <= 1e-9:
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
            if pd.isna(price_local) or price_local <= 1e-9:
                continue

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


@profile
def extract_realized_capital_gains_history(
    all_transactions_df: pd.DataFrame,
    display_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    shortable_symbols: Set[str],
    stock_quantity_close_tolerance: float = STOCK_QUANTITY_CLOSE_TOLERANCE,
    include_accounts: Optional[List[str]] = None,
    current_fx_rates_vs_usd: Optional[Dict[str, float]] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Calculates realized capital gains from transactions using FIFO accounting for long positions.
    Wrapper around calculate_fifo_lots_and_gains.
    """
    logging.info(
        f"Extracting realized capital gains history for display in {display_currency}..."
    )

    if not isinstance(all_transactions_df, pd.DataFrame) or all_transactions_df.empty:
        logging.warning("Capital gains: Input transactions DataFrame is empty.")
        # Return empty DF with correct columns (logic inside calc fn handles this but we need empty input handling)
        return pd.DataFrame(
             columns=[
                "Date", "Symbol", "Account", "Type", "Quantity", 
                "Avg Sale Price (Local)", "Total Proceeds (Local)", 
                "Total Cost Basis (Local)", "Realized Gain (Local)", 
                "Sale/Cover FX Rate", "Total Proceeds (Display)", 
                "Total Cost Basis (Display)", "Realized Gain (Display)", 
                "LocalCurrency", "original_tx_id"
            ]
        )

    # 1. Prepare transactions (filtering logic moved here to keep wrapper clean or passed down)
    # The original logic filtered AFTER calculation. We will follow that for compatibility.
    # But we MUST process ALL accounts for the calculation itself.
    transactions_to_process = all_transactions_df.copy()
    transactions_to_process.sort_values(by=["Date", "original_index"], inplace=True)

    # 2. Call core logic
    df_gains, _ = calculate_fifo_lots_and_gains(
        transactions_df=transactions_to_process,
        display_currency=display_currency,
        historical_fx_yf=historical_fx_yf,
        default_currency=default_currency,
        shortable_symbols=shortable_symbols,
        stock_quantity_close_tolerance=stock_quantity_close_tolerance,
        current_fx_rates_vs_usd=current_fx_rates_vs_usd,
    )

    # 3. Filter accounts (User View Filter)
    if include_accounts and isinstance(include_accounts, list):
        include_accounts_upper = [str(acc).upper().strip() for acc in include_accounts]
        if not df_gains.empty:
             df_gains = df_gains[df_gains["Account"].isin(include_accounts_upper)]

    # 4. Filter by Date (Tax Year / Custom Range)
    if not df_gains.empty:
        # Ensure Date column is datetime or date
        # It usually comes out as datetime from calculate_fifo_lots_and_gains (from transactions_df)
        # normalize to date for comparison
        if "Date" in df_gains.columns:
             # Convert to datetime if needed, though likely already is
             if not pd.api.types.is_datetime64_any_dtype(df_gains["Date"]):
                 df_gains["Date"] = pd.to_datetime(df_gains["Date"])
             
             if from_date:
                 # pd.Timestamp(from_date) compares well with datetime64
                 df_gains = df_gains[df_gains["Date"] >= pd.Timestamp(from_date)]
             
             if to_date:
                 df_gains = df_gains[df_gains["Date"] <= pd.Timestamp(to_date)]

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

# --- Portfolio Health Analysis ---

def calculate_hhi(weights: List[float]) -> float:
    """
    Calculates the Herfindahl-Hirschman Index (HHI) for a list of weights.
    HHI = sum(w_i^2) where w_i are fractions (0 to 1).
    Returns value between 1/N (perfectly diversified) and 1.0 (concentrated).
    """
    if not weights:
        return 1.0
    
    # Ensure weights are normalized sum to 1
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
        
    normalized_weights = [w / total_w for w in weights]
    hhi = sum(w**2 for w in normalized_weights)
    return hhi

def calculate_health_score(
    summary_df: pd.DataFrame, 
    risk_metrics: Dict[str, float],
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Calculates a comprehensive 'Health Score' for the portfolio.
    
    Components:
    1. Diversification (HHI of asset weights).
    2. Efficiency (Sharpe Ratio).
    3. Volatility Check (Penalty if too high).
    """
    score_breakdown = {}
    
    # 1. Diversification Score (0-100)
    # HHI Approach: Lower is better. 
    # HHI < 1500 (0.15) is Competitive (Good)
    # HHI 1500-2500 (0.15-0.25) is Moderately Concentrated
    # HHI > 2500 (0.25) is Highly Concentrated
    
    relevant_holdings = pd.DataFrame()
    
    # Identify the Market Value column (it usually has currency like "Market Value (USD)")
    mv_col = next((c for c in summary_df.columns if c.startswith("Market Value (")), None)
    
    if not summary_df.empty and 'Symbol' in summary_df.columns and mv_col:
        # Robustly handle is_total if it doesn't exist
        if 'is_total' in summary_df.columns:
            is_total_mask = summary_df['is_total'].fillna(False)
        else:
            is_total_mask = pd.Series(False, index=summary_df.index)

        relevant_holdings = summary_df[
            (summary_df['Symbol'] != 'Total') & 
            (summary_df[mv_col] > 0) & 
            (~is_total_mask)
        ]
    
    if relevant_holdings.empty:
        return {
            "overall_score": 0,
            "components": {},
            "rating": "N/A"
        }
        
    weights = relevant_holdings[mv_col].tolist()
    hhi = calculate_hhi(weights)
    
    # Convert HHI to Score (Inverse). 
    # Current HHI range is [1/N to 1.0].
    # Logic: 
    # HHI 0.1 (10 equal stocks) -> 90 Score
    # HHI 0.2 (5 equal stocks) -> 70 Score
    # HHI 1.0 (1 stock) -> 10 Score
    # Formula: 100 * (1 - hhi^0.4) approx? 
    # 0.1^0.4 = 0.39 -> 100*(1-0.39)=61. Too low.
    # Let's use: 100 * (1.0 - (hhi ** 0.5)) but shift it.
    # Actually, a simple linear-ish curve or adjusted power:
    div_score = 100 * (1.0 - (hhi ** 0.6)) + 10
    div_score = max(0, min(100, div_score))
    
    # 2. Efficiency Score (Sharpe)
    # New Logic:
    # Sharpe 0.0 -> Score 30 (Poor, matching risk-free isn't 'fair' for stocks)
    # Sharpe 1.0 -> Score 80 (Good)
    # Sharpe 1.5 -> Score 100 (Excellent)
    sharpe = risk_metrics.get("Sharpe Ratio", 0.0)
    if sharpe <= 0:
        eff_score = max(0, 30 + (sharpe * 10)) # Penalize negative sharpe
    else:
        # Scale 0 to 1.0 -> 30 to 80
        # Scale 1.0 to 1.5 -> 80 to 100
        if sharpe <= 1.0:
            eff_score = 30 + (sharpe * 50)
        else:
            eff_score = 80 + ((sharpe - 1.0) * 40)
            
    eff_score = max(0, min(100, eff_score))
    
    # 3. Volatility Penalty / Score
    # Volatility logic:
    # Vol 0.0 -> Score 50 (Neutral/Insufficient Data)
    # Vol 10-20% -> Score 100 (Healthy active range)
    # Vol > 30% -> Penalty
    vol = risk_metrics.get("Volatility (Ann.)", 0.0)
    if vol == 0:
        vol_score = 50.0
    elif vol < 0.05:
        vol_score = 70.0 # Extreme low vol (cash-like)
    elif vol <= 0.25:
        vol_score = 100.0 # Ideal range
    else:
        # Penalty for high vol
        vol_score = max(0, 100 - (vol - 0.25) * 200) # 35% vol -> 80 score
        
    # Composite Score
    # Weighted average: Div 40%, Eff 40%, Vol 20%
    overall = (div_score * 0.4) + (eff_score * 0.4) + (vol_score * 0.2)
    
    # Rating
    if overall >= 80: rating = "Excellent"
    elif overall >= 60: rating = "Good"
    elif overall >= 40: rating = "Fair"
    elif overall >= 20: rating = "Poor"
    else: rating = "Critical"
    
    return {
        "overall_score": round(overall, 1),
        "rating": rating,
        "components": {
            "diversification": {
                "score": round(div_score, 1),
                "metric": round(hhi, 3), 
                "label": "HHI (Concentration)"
            },
            "efficiency": {
                "score": round(eff_score, 1),
                "metric": round(sharpe, 2),
                "label": "Sharpe Ratio"
            },
            "stability": {
                "score": round(vol_score, 1),
                "metric": f"{round(vol * 100, 1)}%",
                "label": "Volatility"
            }
        }
    }
