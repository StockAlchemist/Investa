"""Daily net external cash-flow calculation (vectorized + reference implementation).

Split from portfolio_logic.py.
"""

# ruff: noqa: E402
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CASH_SYMBOL_CSV
from finutils import get_historical_rate_via_usd_bridge, is_cash_symbol

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


def _calculate_daily_net_cash_flow_vectorized(
    date_range: pd.DatetimeIndex,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,
    internal_to_yf_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.Series, bool]:
    """
    Vectorized calculation of daily net external cash flow (deposits/withdrawals + transfers).
    Returns a Series indexed by Date with the net flow in target_currency.
    """
    # Initialize result series (0.0 for all days)
    daily_net_flow = pd.Series(0.0, index=date_range)
    has_lookup_errors = False
    
    if transactions_df.empty:
        return daily_net_flow, False

    # Filter transactions up to the end date (Capture historical flows for 'Initial Flow')
    # FIX: Ensure transaction dates are UTC for comparison with UTC date_range
    tx_dates = pd.to_datetime(transactions_df["Date"], utc=True)
    mask_date = (tx_dates <= date_range[-1])
    df_period = transactions_df[mask_date].copy()
    
    if df_period.empty:
        return daily_net_flow, False

    # --- PROACTIVE TIMEZONE NORMALIZATION ---
    # Ensure transaction dates match the target date_range timezone EXACTLY.
    # This prevents "tz-naive vs tz-aware" errors during later aggregation.
    target_tz = date_range.tz
    
    # 1. Convert to datetime and FORCIBLY NORMALIZE to naive to avoid mixed-awareness object dtypes
    # This is critical for Transfers which often have timezones from external exports.
    df_period["Date"] = pd.to_datetime(df_period["Date"], utc=True).dt.tz_localize(None)
    
    # 2. Align Timezone (Now that it's clean and naive)
    if target_tz is not None:
        # Target is Aware
        df_period["Date"] = df_period["Date"].dt.tz_localize(target_tz)
    # else: Already naive from the normalization above

    # Normalize included accounts
    included_set = set()
    if included_accounts:
        included_set = {str(a).upper().strip() for a in included_accounts}
    
    # --- 1. CASH SYMBOL FLOWS (Deposits/Withdrawals) ---
    # Use the centralized bookkeeping-mode-independent classifier from
    # finutils. This collapses three previously divergent rules across the TWR
    # engine, the IRR engine, and ad-hoc checks into one definition: an
    # external flow is a $CASH Deposit/Withdrawal whose Note does NOT begin
    # with "Auto-generated:" (the import tool's marker for per-trade synthetic
    # entries). $CASH buy/sell, $CASH dividend/interest, and stock-symbol
    # buy/sell are all internal under this classifier.
    from finutils import compute_external_flow_mask
    cash_mask = compute_external_flow_mask(df_period)
    df_cash = df_period[cash_mask].copy()

    # --- FIX: Filter cash flows by included accounts ---
    if included_accounts and not df_cash.empty:
        df_cash = df_cash[
            df_cash["Account"].astype(str).str.upper().str.strip().isin(included_set)
        ]

    
    flows_list = [] # Store partial dataframes (Date, Flow_Local, Currency)

    if not df_cash.empty:
        # Vectorized calculation of local flow
        total_amount_col = df_cash["Total Amount"] if "Total Amount" in df_cash.columns else pd.Series(0.0, index=df_cash.index)
        total_amount = pd.to_numeric(total_amount_col, errors="coerce").fillna(0.0).abs()
        qty = pd.to_numeric(df_cash["Quantity"], errors="coerce").fillna(0.0).abs()
        comm = pd.to_numeric(df_cash["Commission"], errors="coerce").fillna(0.0)
        
        # Prefer Total Amount if non-zero, otherwise fall back to Quantity
        cash_amt = np.where(total_amount > 1e-9, total_amount, qty)
        
        # 1. Deposit
        is_dep = df_cash["Type"].str.lower() == "deposit"
        # 2. Withdrawal
        is_wd = df_cash["Type"].str.lower() == "withdrawal"
        
        local_flow = pd.Series(0.0, index=df_cash.index)
        local_flow[is_dep] = cash_amt[is_dep] - comm[is_dep]
        local_flow[is_wd] = -cash_amt[is_wd] - comm[is_wd]
        
        df_cash["_flow_local"] = local_flow
        flows_list.append(df_cash[["Date", "_flow_local", "Local Currency"]])

    # --- 2. ASSET TRANSFERS (In/Out) ---
    if included_accounts:
        # Transfers are flows if they cross the boundary of "included_accounts".
        # This MUST include both Assets and $CASH transfers.
        # --- NEW: Include Cleanup/Adjustment transactions as flows ---
        # These are ledger corrections that should not affect portfolio performance metrics.
        # Treating them as external flows (like transfers) neutralizes their impact on TWR.
        cleanup_keywords = ["cleanup", "dust", "ghost", "artifact", "adjustment", "correction"]
        note_col = "Note" if "Note" in df_period.columns else None
        cleanup_mask = pd.Series(False, index=df_period.index)
        if note_col:
            cleanup_mask = df_period[note_col].fillna("").str.lower().apply(
                lambda x: any(kw in x for kw in cleanup_keywords)
            )
        
        # Transfers and Cleanup/Adjustments are both treated as boundary-crossing flows
        trans_mask = (df_period["Type"].str.lower().str.strip() == "transfer") | cleanup_mask
        df_trans = df_period[trans_mask].copy()
                
        if not df_trans.empty:
            # Determine direction multiplier
            src_in = df_trans["Account"].astype(str).str.upper().str.strip().isin(included_set)
            dest_in = pd.Series(False, index=df_trans.index)
            if "To Account" in df_trans.columns:
                 dest_in = df_trans["To Account"].astype(str).str.upper().str.strip().isin(included_set)
            
            # 1. Standard Transfer Multiplier: src_in & ~dest_in -> -1 (OUT); ~src_in & dest_in -> +1 (IN)
            multiplier = pd.Series(0.0, index=df_trans.index)
            is_real_transfer = df_trans["Type"].str.lower().str.strip() == "transfer"
            
            multiplier[is_real_transfer & src_in & ~dest_in] = -1.0
            multiplier[is_real_transfer & ~src_in & dest_in] = 1.0
            
            # 2. Cleanup Multiplier: Buy -> +1 (IN); Sell/Withdrawal -> -1 (OUT)
            # This ensures that fixing ledger artifacts doesn't create fake performance gains/losses.
            is_cleanup = cleanup_mask.reindex(df_trans.index).fillna(False) & ~is_real_transfer
            if is_cleanup.any():
                type_l = df_trans["Type"].str.lower().str.strip()
                multiplier[is_cleanup & type_l.isin(["buy", "deposit"])] = 1.0
                multiplier[is_cleanup & type_l.isin(["sell", "withdrawal"])] = -1.0
                        
            # Filter only relevant transactions (those that have a non-zero impact on the portfolio boundary)
            relevant_trans = df_trans[multiplier != 0.0].copy()
            multiplier = multiplier[multiplier != 0.0]

            
            if not relevant_trans.empty:
                # Calculate Value: Qty * Price
                qty = pd.to_numeric(relevant_trans["Quantity"], errors="coerce").fillna(0.0).abs()
                price = pd.to_numeric(relevant_trans["Price/Share"], errors="coerce").fillna(0.0)
                # Ensure $CASH always has price 1.0
                is_cash_mask = relevant_trans["Symbol"].str.upper() == CASH_SYMBOL_CSV.upper()
                price[is_cash_mask] = 1.0
                                
                # FALLBACK PRICE LOOKUP (Vectorized-ish)
                missing_price_mask = (price <= 1e-9)
                
                if missing_price_mask.any():
                    relevant_trans_missing = relevant_trans[missing_price_mask]
                    unique_syms = relevant_trans_missing["Symbol"].unique()

                    for sym in unique_syms:
                        # Build sym_mask over the FULL relevant_trans index (not just
                        # the missing-subset) so it aligns with `price` for boolean
                        # indexing. The "& missing_price_mask" restricts assignment
                        # to rows that actually need a fallback price.
                        sym_mask = (relevant_trans["Symbol"] == sym) & missing_price_mask
                        price_found = 0.0

                        # Fallback A: Yahoo Finance Lookup
                        yf_sym = internal_to_yf_map.get(sym) if internal_to_yf_map else None
                        if yf_sym and historical_prices_yf_unadjusted and yf_sym in historical_prices_yf_unadjusted:
                            try:
                                price_series = historical_prices_yf_unadjusted[yf_sym]["price"]
                                if not price_series.empty:
                                    # Ensure price_series is sorted for asof
                                    if not price_series.index.is_monotonic_increasing:
                                        price_series = price_series.sort_index()
                                    if not isinstance(price_series.index, pd.DatetimeIndex):
                                        price_series.index = pd.to_datetime(price_series.index)

                                    # Forward fill to get a price for the transfer date
                                    dates_needed = pd.to_datetime(relevant_trans.loc[sym_mask, "Date"])
                                    # Use reindex with nearest or ffill
                                    relevant_prices = price_series.reindex(dates_needed, method='ffill')
                                    price_found = relevant_prices.iloc[0] # Take first for simplicity in this loop
                            except Exception as e:
                                logging.warning(f"Transfer price fallback failed for {sym}: {e}")

                        # Fallback B: Any other transaction price in history
                        if price_found <= 1e-9:
                            match_tx = df_period[(df_period["Symbol"] == sym) & (df_period["Price/Share"] > 1e-9)]
                            if not match_tx.empty:
                                price_found = match_tx.iloc[0]["Price/Share"]

                        if price_found > 1e-9:
                            price.loc[sym_mask] = price_found
                        else:
                            # Final Fail: Log for visibility
                            try:
                                with open("/Users/kmatan/.gemini/antigravity/brain/19ed956f-9dc7-40d0-85b1-1aa49c84c1d9/scratch/missing_transfer_prices.txt", "a") as f:
                                    f.write(f"Missing price for transfer: {sym}\n")
                            except OSError:
                                pass

                value = qty * price
                local_flow = value * multiplier
                
                relevant_trans["_flow_local"] = local_flow
                flows_list.append(relevant_trans[["Date", "_flow_local", "Local Currency"]])

    if not flows_list:
        return daily_net_flow, False

    # --- 3. CONSOLIDATE AND FX CONVERT ---
    all_flows = pd.concat(flows_list)
    flows_grouped = all_flows.groupby(["Date", "Local Currency"])["_flow_local"].sum().reset_index()
    
    unique_currs = flows_grouped["Local Currency"].unique()
    final_flows = []
    
    for curr in unique_currs:
        mask_c = flows_grouped["Local Currency"] == curr
        sub_df = flows_grouped[mask_c].copy()
        
        if curr == target_currency:
             final_flows.append(sub_df.set_index("Date")["_flow_local"])
        else:
             # FIX: Robust Vectorized Lookup using Continuous Series
             # Instead of picking single points which might fail or differ from valuation,
             # we construct the full cross-rate series for the date_range and lookup.
             
             # 1. Build Target Rate Series (Target/USD)
             target_rate_s = None
             if target_currency.upper() == "USD":
                 target_rate_s = pd.Series(1.0, index=date_range)
             else:
                 t_pair = f"{target_currency.upper()}=X"
                 if t_pair in historical_fx_yf:
                      t_df = historical_fx_yf[t_pair]
                      if not t_df.empty and "price" in t_df.columns:
                           t_s = t_df["price"]
                           t_s.index = pd.to_datetime(t_s.index, utc=True)
                           # AGGRESSIVE BACKFILL
                           target_rate_s = t_s.reindex(date_range).ffill().bfill()
             if target_rate_s is None:
                  target_rate_s = pd.Series(np.nan, index=date_range)

             # 2. Build Local Rate Series (Local/USD)
             local_rate_s = None
             if curr.upper() == "USD":
                 local_rate_s = pd.Series(1.0, index=date_range)
             else:
                 l_pair = f"{curr.upper()}=X"
                 if l_pair in historical_fx_yf:
                      l_df = historical_fx_yf[l_pair]
                      if not l_df.empty and "price" in l_df.columns:
                           l_s = l_df["price"]
                           l_s.index = pd.to_datetime(l_s.index, utc=True)
                           # AGGRESSIVE BACKFILL
                           local_rate_s = l_s.reindex(date_range).ffill().bfill()
             if local_rate_s is None:
                  local_rate_s = pd.Series(np.nan, index=date_range)
             
             # 3. Compute Cross Rate Series
             with np.errstate(divide='ignore', invalid='ignore'):
                 cross_rate_s = target_rate_s / local_rate_s
                 # Fill any remaining NaNs (if one series was totally missing)
                 if cross_rate_s.isna().any():
                      cross_rate_s = cross_rate_s.ffill().bfill()
                      # Final fallback to 1.0 ONLY if totally empty (catastrophic)
                      cross_rate_s = cross_rate_s.fillna(1.0)

             # 4. Map transactions to this series
             # Transactions are "Date" (naive or aware). 
             # Series is date_range (likely aware UTC).
             # We need to map safely. index of cross_rate_s is DatetimeIndex.
             
             # Create a lookup map: Date(date) -> Rate
             # This is fast enough for <10k days.
             daily_rate_map = pd.Series(cross_rate_s.values, index=cross_rate_s.index.date).to_dict()
             
             factors = sub_df["Date"].dt.date.map(daily_rate_map)
             
             # Check for unmapped?
             if factors.isna().any():
                 # Should not happen with bfilled series unless date out of range?
                 # If date out of range (snapping might put it inside range?), fallback to series mean or 1.0
                 factors = factors.fillna(1.0) 
                 has_lookup_errors = True

             converted = sub_df["_flow_local"] * factors
             converted.index = sub_df["Date"]
             final_flows.append(converted)

    if final_flows:
        total_series = pd.concat(final_flows)
        daily_net_flow_add = total_series.groupby(level=0).sum()
        
        # DYNAMIC TIMEZONE ALIGNMENT
        # Align `daily_net_flow_add` (derived from transactions, originally naive) 
        # to match `daily_net_flow` (which comes from date_range, typically UTC).
        if isinstance(daily_net_flow.index, pd.DatetimeIndex):
            target_tz = daily_net_flow.index.tz
            
            if isinstance(daily_net_flow_add.index, pd.DatetimeIndex):
                source_tz = daily_net_flow_add.index.tz
                
                if target_tz is not None:
                    # Target is Aware (e.g. UTC)
                    if source_tz is None:
                        # Source is Naive -> Localize to Target
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_localize(target_tz)
                    else:
                        # Source is Aware -> Convert to Target
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_convert(target_tz)
                else:
                    # Target is Naive
                    if source_tz is not None:
                        # Source is Aware -> Make Naive
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_localize(None)

        # SNAPPING: Aggregate any transactions occurring BEFORE the start date into a single 'Initial Flow' 
        # slot on the first day of the range. This ensures Cumulative Net Flow (COST) is accurate 
        # and prevents 0-base TWR spikes.
        start_ts = date_range[0]
        pre_range_mask = daily_net_flow_add.index < start_ts
        if pre_range_mask.any():
            initial_flow_total = daily_net_flow_add[pre_range_mask].sum()
            logging.info(f"STABILIZATION: Found {pre_range_mask.sum()} transactions before start date. Adding Initial Flow of {initial_flow_total:.2f} to {start_ts}")
            
            # Remove the old entries and add back to the first slot
            daily_net_flow_add = daily_net_flow_add[~pre_range_mask]
            if start_ts in daily_net_flow_add.index:
                daily_net_flow_add[start_ts] += initial_flow_total
            else:
                daily_net_flow_add[start_ts] = initial_flow_total
             
        daily_net_flow = daily_net_flow.add(daily_net_flow_add, fill_value=0.0)
    
    return daily_net_flow, has_lookup_errors


def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,  # ADDED
    internal_to_yf_map: Optional[Dict[str, str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the net external cash flow for a specific date in the target currency.
    Now includes ASSET TRANSFERS in/out of the included accounts as flows.
    """
    # --- 1. Filter Transactions for the Date ---
    # Filter for transactions on this specific date
    daily_tx = transactions_df[transactions_df["Date"].dt.date == target_date]
    
    if daily_tx.empty:
        return 0.0, False

    fx_lookup_failed = False
    net_flow_target_curr = 0.0
    
    # 1. External Flows (Deposit/Withdrawal)
    external_flow_types = ["deposit", "withdrawal"]
    
    # Capture ALL deposits and withdrawals, now including non-cash assets
    external_flow_tx = daily_tx[
        (daily_tx["Type"].str.lower().str.strip().isin(external_flow_types))
    ].copy()

    # Process External Flows
    if not external_flow_tx.empty:
        for _, row in external_flow_tx.iterrows():
            tx_type = row["Type"].lower().strip()
            symbol = row["Symbol"]
            is_cash = is_cash_symbol(symbol)
            qty = pd.to_numeric(row["Quantity"], errors="coerce")
            commission_local_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission_local = (
                0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
            )
            local_currency = row["Local Currency"]
            flow_local = 0.0

            if pd.isna(qty):
                continue

            if is_cash:
                if tx_type == "deposit":
                    flow_local = abs(qty) - commission_local
                elif tx_type == "withdrawal":
                    flow_local = -abs(qty) - commission_local
            else:
                # Non-cash Asset Contribution/Removal (e.g., Stock Deposit)
                price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
                
                # If price is missing or zero, try market price lookup
                if pd.isna(price_local) or price_local <= 0:
                    if historical_prices_yf_unadjusted and internal_to_yf_map:
                        yf_ticker = internal_to_yf_map.get(symbol)
                        if yf_ticker and yf_ticker in historical_prices_yf_unadjusted:
                            prices_df = historical_prices_yf_unadjusted[yf_ticker]
                            # Try exact date match first
                            if target_date in prices_df.index:
                                for col_name in ["Close", "price", "Adj Close"]:
                                    if col_name in prices_df.columns:
                                        price_local = prices_df.loc[target_date, col_name]
                                        break
                                else:
                                    if not prices_df.empty:
                                        price_local = prices_df.loc[target_date].iloc[0]
                            else:
                                # Try converting to timestamp
                                try:
                                    target_ts = pd.Timestamp(target_date)
                                    if target_ts in prices_df.index:
                                        for col_name in ["Close", "price", "Adj Close"]:
                                            if col_name in prices_df.columns:
                                                price_local = prices_df.loc[target_ts, col_name]
                                                break
                                        else:
                                            if not prices_df.empty:
                                                price_local = prices_df.loc[target_ts].iloc[0]
                                except (KeyError, IndexError, ValueError):
                                    pass

                if not pd.isna(price_local) and price_local > 0:
                    if tx_type == "deposit":
                        flow_local = (abs(qty) * price_local) - commission_local
                    elif tx_type == "withdrawal":
                        flow_local = -(abs(qty) * price_local) - commission_local
                else:
                    logging.warning(f"Flow: Could not determine price for asset flow {symbol} on {target_date}. Skipping.")
                    continue

            flow_target = flow_local
            if local_currency != target_currency:
                fx_rate = get_historical_rate_via_usd_bridge(
                    local_currency, target_currency, target_date, historical_fx_yf
                )
                if pd.isna(fx_rate):
                    logging.warning(
                        f"Hist FX Lookup CRITICAL Failure for cash flow: {local_currency}->{target_currency} on {target_date}. Aborting day's flow."
                    )
                    fx_lookup_failed = True
                    net_flow_target_curr = np.nan
                    break
                else:
                    flow_target = flow_local * fx_rate

            if pd.isna(net_flow_target_curr):
                break
            if pd.notna(flow_target):
                net_flow_target_curr += flow_target
            else:
                net_flow_target_curr = np.nan
                fx_lookup_failed = True
                break

    if fx_lookup_failed:
        return np.nan, True

    # 2. Asset Transfer Flows (Transfer IN/OUT)
    if included_accounts:
        # Normalize included_accounts to uppercase/stripped
        included_set = {acc.strip().upper() for acc in included_accounts}
        transfer_tx = daily_tx[daily_tx["Type"].str.lower().str.strip() == "transfer"].copy()
        
        for _, row in transfer_tx.iterrows():
            symbol = row["Symbol"]
            is_cash = is_cash_symbol(symbol)
            # FIX: Do NOT skip cash transfers if we are in specific account view.
            # Cash transfers betwen accounts ARE external flows for the specific account.
            # if is_cash_symbol(symbol): # Skip cash transfers (handled by cash balance logic usually, or ignored if internal)
            #      continue
                 
            account = row["Account"]
            to_account = row.get("To Account")
            qty = pd.to_numeric(row["Quantity"], errors="coerce")
            
            # Determine Price/Value of the transfer
            price_local = 1.0 # Default for cash
            if not is_cash:
                price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
            
            local_currency = row["Local Currency"]
            
            if pd.isna(qty):
                continue

            # FIX: If price is missing or 0 (for non-cash), try to look up market price
            if not is_cash and (pd.isna(price_local) or price_local <= 0):
                if historical_prices_yf_unadjusted and internal_to_yf_map:
                    yf_ticker = internal_to_yf_map.get(symbol)
                    if yf_ticker and yf_ticker in historical_prices_yf_unadjusted:
                        prices_df = historical_prices_yf_unadjusted[yf_ticker]
                        
                        # Look for price on target_date
                        # Try exact match first
                        if target_date in prices_df.index:
                            if "Close" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "Close"]
                            elif "price" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "price"]
                            elif "Adj Close" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "Adj Close"]
                            elif not prices_df.empty:
                                # Fallback to first column if standard names missing
                                price_local = prices_df.loc[target_date].iloc[0]
                        else:
                            # Try converting target_date to datetime if index is datetime
                            try:
                                target_ts = pd.Timestamp(target_date)
                                if target_ts in prices_df.index:
                                    if "Close" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "Close"]
                                    elif "price" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "price"]
                                    elif "Adj Close" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "Adj Close"]
                                    elif not prices_df.empty:
                                        price_local = prices_df.loc[target_ts].iloc[0]
                            except (KeyError, IndexError, ValueError):
                                pass
                                
            if not is_cash and (pd.isna(price_local) or price_local <= 0):
                continue
                
            # Normalize account names from transaction
            account_norm = str(account).strip().upper()
            to_account_norm = str(to_account).strip().upper() if to_account else None
            
            is_from_included = account_norm in included_set
            is_to_included = to_account_norm in included_set if to_account_norm else False
            
            flow_val_local = 0.0
            
            # Determine Value of flow (Qty * Price)
            # For cash, price is 1.0 (local currency)
            transfer_value = abs(qty) * price_local
            
            if is_from_included and not is_to_included:
                # Transfer OUT (Withdrawal) onto the included set
                # e.g. From "MyAcct" (Included) -> To "External" (Excluded)
                flow_val_local = -transfer_value
            elif not is_from_included and is_to_included:
                # Transfer IN (Deposit) onto the included set
                # e.g. From "External" (Excluded) -> To "MyAcct" (Included)
                flow_val_local = transfer_value
            # Else: Internal Transfer (Included -> Included) or External (Excluded -> Excluded) -> Net Flow 0
            # (Internal transfers cancel out for the group, so 0 net external flow)
            
            if abs(flow_val_local) > 1e-9:
                flow_target = flow_val_local
                if local_currency != target_currency:
                    fx_rate = get_historical_rate_via_usd_bridge(
                        local_currency, target_currency, target_date, historical_fx_yf
                    )
                    if pd.isna(fx_rate):
                        fx_lookup_failed = True
                        net_flow_target_curr = np.nan
                        break
                    flow_target = flow_val_local * fx_rate
                
                if pd.notna(flow_target):
                    net_flow_target_curr += flow_target

    return net_flow_target_curr, fx_lookup_failed
