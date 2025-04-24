# --- START OF MODIFIED portfolio_logic.py ---
# --- Imports needed within this function's scope or globally ---
import pandas as pd
from datetime import datetime, date, timedelta, UTC # Add UTC here
import requests # Keep for potential future use or different APIs
import os
import json
import numpy as np
from scipy import optimize
from typing import List, Tuple, Dict, Optional, Any, Set
import traceback # For detailed error logging
from collections import defaultdict
import time # For adding slight delays if needed (historical fetch)
from io import StringIO # For historical cache loading
# --- Multiprocessing Imports ---
import multiprocessing
from functools import partial
# --- End Multiprocessing Imports ---
import calendar # Added for potential market day checks
import hashlib # Added for cache key hashing

# ADD THIS near the start of the file for easy toggling
HISTORICAL_DEBUG_USD_CONVERSION = False # Set to True only when debugging this specific issue
HISTORICAL_DEBUG_SET_VALUE = False # Set to True only when debugging this specific issue
DEBUG_DATE_VALUE = date(2024, 2, 5) # Choose a relevant date within your range where SET should have value

# --- Finance API Import ---
try:
    import yfinance as yf # Import yfinance
    YFINANCE_AVAILABLE = True
except ImportError:
    print("WARNING: yfinance library not found. Stock/FX fetching and historical data will fail.")
    print("         Please install it: pip install yfinance")
    YFINANCE_AVAILABLE = False
    class DummyYFinance:
        def Tickers(self, *args, **kwargs): raise ImportError("yfinance not installed")
        def download(self, *args, **kwargs): raise ImportError("yfinance not installed")
        def Ticker(self, *args, **kwargs): raise ImportError("yfinance not installed")
    yf = DummyYFinance()

# --- Constants ---
CASH_SYMBOL_CSV = '$CASH' # Standardized cash symbol

# --- Caching ---
DEFAULT_CURRENT_CACHE_FILE_PATH = 'portfolio_cache_yf.json'
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = 'yf_portfolio_hist_raw_adjusted_v7'
DAILY_RESULTS_CACHE_PATH_PREFIX = 'yf_portfolio_daily_results_v10' # <-- V10 cache with daily_return & daily_gain
HISTORICAL_RESULTS_CACHE_FILE_PATH = 'historical_results_cache.json' # Legacy cache for summary/IRR results
YFINANCE_CACHE_DURATION_HOURS = 4 # Keep for CURRENT data

# --- Yahoo Finance Mappings & Configuration ---
# (Keep existing mappings and configs: YFINANCE_INDEX_TICKER_MAP, SYMBOL_MAP_TO_YFINANCE, YFINANCE_EXCLUDED_SYMBOLS, SHORTABLE_SYMBOLS, DEFAULT_CURRENCY)
YFINANCE_INDEX_TICKER_MAP = { ".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())
SYMBOL_MAP_TO_YFINANCE = { "BRK.B": "BRK-B", "AAPL": "AAPL", "GOOG": "GOOG", "GOOGL": "GOOGL", "MSFT": "MSFT", "AMZN": "AMZN", "LQD": "LQD", "SPY": "SPY", "VTI": "VTI", "KHC": "KHC", "DIA": "DIA", "AXP": "AXP", "BLV": "BLV", "NVDA": "NVDA", "PLTR": "PLTR", "JNJ": "JNJ", "XLE": "XLE", "VDE": "VDE", "BND": "BND", "VWO": "VWO", "DPZ": "DPZ", "QQQ": "QQQ", "BHP": "BHP", "DAL": "DAL", "QSR": "QSR", "ASML": "ASML", "NLY": "NLY", "ADRE": "ADRE", "GS": "GS", "EPP": "EPP", "EFA": "EFA", "IBM": "IBM", "VZ": "VZ", "BBW": "BBW", "CVX": "CVX", "NKE": "NKE", "KO": "KO", "BAC": "BAC", "VGK": "VGK", "C": "C", # Add others...
                          "TLT": "TLT", "AGG": "AGG", "^GSPC": "^GSPC", "VT": "VT", "IWM": "IWM", }
# YFINANCE_EXCLUDED_SYMBOLS = set([ "BBW", "IDBOX", "IDIOX", "ES-Fixed_Income", "GENCO:BKK", "UOBBC", "ES-JUMBO25", "SCBCHA-SSF", "ES-SET50", "ES-Tresury", "UOBCG", "ES-GQG", "SCBRM1", "SCBRMS50", "AMARIN:BKK", "RIMM", "SCBSFF", "BANPU:BKK", "AAV:BKK", "CPF:BKK", "EMV", "IDMOX", "BML:BKK", "ZEN:BKK", "SCBRCTECH", "MBK:BKK", "DSV", "THAI:BKK", "IDLOX", "SCBRMS&P500", "AOT:BKK", "BECL:BKK", "TCAP:BKK", "KRFT", "AAUKY", "NOK:BKK", "ADRE", "SCC:BKK", "CPALL:BKK", "TRUE:BKK", "PTT:BKK", "ES-FIXED_INCOME", "ES-TRESURY", "BEM:BKK" ])
YFINANCE_EXCLUDED_SYMBOLS = set()
SHORTABLE_SYMBOLS = {'AAPL', 'RIMM'} # Used RIMM instead of BB
DEFAULT_CURRENCY = 'USD'

# --- Helper Functions ---

# --- File Hashing Helper ---
def _get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as file:
            while chunk := file.read(8192): # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found for hashing: {filepath}")
        return "FILE_NOT_FOUND"
    except Exception as e:
        print(f"Error hashing file {filepath}: {e}")
        return "HASHING_ERROR"

# (_load_and_clean_transactions, calculate_npv, calculate_irr, get_cash_flows_for_symbol_account,
#  get_cash_flows_for_mwr, fetch_index_quotes_yfinance, get_cached_or_fetch_yfinance_data,
#  get_conversion_rate)
def _load_and_clean_transactions(
    transactions_csv_file: str,
    account_currency_map: Dict, # Now required
    default_currency: str      # Now required
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Set[int], Dict[int, str]]:
    """
    Loads and cleans transactions from CSV, adding a 'Local Currency' column
    based on the provided account_currency_map.
    Returns cleaned_df, original_df, ignored_indices, ignored_reasons.
    """
    original_transactions_df: Optional[pd.DataFrame] = None
    transactions_df: Optional[pd.DataFrame] = None
    ignored_row_indices = set()
    ignored_reasons: Dict[int, str] = {}

    print(f"Helper: Attempting to load transactions from: {transactions_csv_file}")
    try:
        # ... (dtype_spec, na_values, pd.read_csv remain the same) ...
        dtype_spec = { "Quantity of Units": str, "Amount per unit": str, "Total Amount": str, "Fees": str, "Split Ratio (new shares per old share)": str }
        na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        original_transactions_df = pd.read_csv( transactions_csv_file, header=0, skipinitialspace=True, keep_default_na=True, na_values=na_values, dtype=dtype_spec, encoding='utf-8' )
        transactions_df = original_transactions_df.copy()
        print(f"Helper: Successfully loaded {len(transactions_df)} records.")
    except FileNotFoundError: return None, None, ignored_row_indices, ignored_reasons
    except UnicodeDecodeError as e: return None, original_transactions_df, ignored_row_indices, ignored_reasons
    except Exception as e: return None, original_transactions_df, ignored_row_indices, ignored_reasons
    if transactions_df is None or transactions_df.empty: return None, original_transactions_df, ignored_row_indices, ignored_reasons

    try:
        transactions_df['original_index'] = transactions_df.index

        # --- Column Mapping (unchanged) ---
        column_mapping = { "Date (MMM DD, YYYY)": "Date", "Transaction Type": "Type", "Stock / ETF Symbol": "Symbol", "Quantity of Units": "Quantity", "Amount per unit": "Price/Share", "Total Amount": "Total Amount", "Fees": "Commission", "Split Ratio (new shares per old share)": "Split Ratio", "Investment Account": "Account", "Note": "Note" }

        # --- Column Existence Check and Renaming (unchanged) ---
        required_original_cols = [ "Date (MMM DD, YYYY)", "Transaction Type", "Stock / ETF Symbol", "Investment Account" ]
        actual_columns = [col.strip() for col in transactions_df.columns]
        required_stripped_cols = {col: col.strip() for col in required_original_cols}
        missing_original = [orig_col for orig_col, stripped_col in required_stripped_cols.items() if stripped_col not in actual_columns]
        if missing_original: raise ValueError(f"Missing essential CSV columns: {missing_original}. Found columns: {transactions_df.columns.tolist()}")
        rename_dict = {stripped_csv_header: internal_name for csv_header, internal_name in column_mapping.items() if (stripped_csv_header := csv_header.strip()) in actual_columns}
        transactions_df.columns = actual_columns
        transactions_df.rename(columns=rename_dict, inplace=True)

        # --- Basic Cleaning (Symbol, Type - unchanged) ---
        transactions_df['Symbol'] = transactions_df['Symbol'].fillna('UNKNOWN_SYMBOL').astype(str).str.strip().str.upper(); transactions_df.loc[transactions_df['Symbol'] == '', 'Symbol'] = 'UNKNOWN_SYMBOL'
        transactions_df.loc[transactions_df['Symbol'] == '$CASH', 'Symbol'] = CASH_SYMBOL_CSV
        transactions_df['Type'] = transactions_df['Type'].fillna('UNKNOWN_TYPE').astype(str).str.strip().str.lower(); transactions_df.loc[transactions_df['Type'] == '', 'Type'] = 'UNKNOWN_TYPE'

        # --- Clean Account and ADD Local Currency ---
        transactions_df['Account'] = transactions_df['Account'].fillna('Unknown').astype(str).str.strip()
        transactions_df.loc[transactions_df['Account'] == '', 'Account'] = 'Unknown'
        # Map account to currency using the provided map, fall back to default_currency
        transactions_df['Local Currency'] = transactions_df['Account'].map(account_currency_map).fillna(default_currency)
        print(f"Helper: Added 'Local Currency' column. Example: {transactions_df[['Account', 'Local Currency']].head().to_string(index=False)}")
        # --- End Local Currency Addition ---

        # --- Date Parsing (unchanged) ---
        transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], errors='coerce')
        # ... (fallback date parsing logic remains the same) ...
        if transactions_df['Date'].isnull().any():
             formats_to_try = ['%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%b-%Y', '%Y%m%d']
             for fmt in formats_to_try:
                  if transactions_df['Date'].isnull().any():
                      nat_mask = transactions_df['Date'].isnull()
                      try: parsed = pd.to_datetime(transactions_df.loc[nat_mask, 'Date'].astype(str), format=fmt, errors='coerce'); transactions_df.loc[nat_mask, 'Date'] = parsed
                      except Exception: pass
             if transactions_df['Date'].isnull().any():
                 nat_mask = transactions_df['Date'].isnull()
                 try: inferred = pd.to_datetime(transactions_df.loc[nat_mask, 'Date'], infer_datetime_format=True, errors='coerce'); transactions_df.loc[nat_mask, 'Date'] = inferred
                 except Exception: pass
        bad_date_indices = transactions_df[transactions_df['Date'].isnull()].index
        if not bad_date_indices.empty:
             for idx in transactions_df.loc[bad_date_indices, 'original_index']: ignored_reasons[idx] = "Invalid/Unparseable Date"
             ignored_row_indices.update(transactions_df.loc[bad_date_indices, 'original_index'])
             transactions_df.drop(bad_date_indices, inplace=True)
        if transactions_df.empty:
             print("Helper WARN: All rows dropped due to invalid dates.")
             return None, original_transactions_df, ignored_row_indices, ignored_reasons

        # --- Numeric Conversion (unchanged) ---
        numeric_cols = ['Quantity', 'Price/Share', 'Total Amount', 'Commission', 'Split Ratio']
        for col in numeric_cols:
            if col in transactions_df.columns:
                if transactions_df[col].dtype == 'object': transactions_df[col] = transactions_df[col].astype(str).str.replace(',', '', regex=False)
                transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')

        # --- Flagging Rows to Drop (unchanged) ---
        initial_row_count = len(transactions_df); rows_to_drop_indices = pd.Index([])
        def flag_for_drop(indices, reason):
            if not indices.empty:
                original_indices = transactions_df.loc[indices, 'original_index'].tolist()
                for orig_idx in original_indices: ignored_reasons[orig_idx] = reason
            return rows_to_drop_indices.union(indices)
        # ... (rest of flagging logic remains the same) ...
        is_buy_sell_stock = transactions_df['Type'].isin(['buy', 'sell', 'deposit', 'withdrawal']) & (transactions_df['Symbol'] != CASH_SYMBOL_CSV)
        is_short_stock = transactions_df['Type'].isin(['short sell', 'buy to cover']) & transactions_df['Symbol'].isin(SHORTABLE_SYMBOLS)
        is_split = transactions_df['Type'].isin(['split', 'stock split']); is_dividend = transactions_df['Type'] == 'dividend'; is_fees = transactions_df['Type'] == 'fees'
        is_cash_tx = (transactions_df['Symbol'] == CASH_SYMBOL_CSV) & transactions_df['Type'].isin(['buy', 'sell', 'deposit', 'withdrawal'])
        nan_qty_or_price = transactions_df[['Quantity', 'Price/Share']].isnull().any(axis=1)
        idx = transactions_df.index[is_buy_sell_stock & nan_qty_or_price]; rows_to_drop_indices = flag_for_drop(idx, "Missing Qty/Price Stock")
        idx = transactions_df.index[is_short_stock & nan_qty_or_price]; rows_to_drop_indices = flag_for_drop(idx, "Missing Qty/Price Short")
        idx = transactions_df.index[is_cash_tx & transactions_df['Quantity'].isnull()]; rows_to_drop_indices = flag_for_drop(idx, "Missing $CASH Qty")
        invalid_split = transactions_df['Split Ratio'].isnull() | (transactions_df['Split Ratio'] <= 0); idx = transactions_df.index[is_split & invalid_split]; rows_to_drop_indices = flag_for_drop(idx, "Missing/Invalid Split Ratio")
        missing_div = transactions_df['Total Amount'].isnull() & transactions_df['Price/Share'].isnull(); idx = transactions_df.index[is_dividend & missing_div]; rows_to_drop_indices = flag_for_drop(idx, "Missing Dividend Amt/Price")
        idx = transactions_df.index[is_fees & transactions_df['Commission'].isnull()]; rows_to_drop_indices = flag_for_drop(idx, "Missing Fee Commission")
        transactions_df['Commission'] = transactions_df['Commission'].fillna(0.0)
        is_unknown = (transactions_df['Symbol'] == 'UNKNOWN_SYMBOL') | (transactions_df['Type'] == 'UNKNOWN_TYPE'); idx = transactions_df.index[is_unknown].difference(rows_to_drop_indices); rows_to_drop_indices = flag_for_drop(idx, "Unknown Symbol/Type")

        if not rows_to_drop_indices.empty:
            original_indices_to_ignore = transactions_df.loc[rows_to_drop_indices, 'original_index'].tolist(); ignored_row_indices.update(original_indices_to_ignore)
            transactions_df.drop(rows_to_drop_indices, inplace=True);
        if transactions_df.empty:
             print("Helper WARN: All transactions dropped during cleaning.")
             return None, original_transactions_df, ignored_row_indices, ignored_reasons

        # --- Sort and Return (unchanged) ---
        transactions_df.sort_values(by=['Date', 'original_index'], inplace=True, ascending=True)
        return transactions_df, original_transactions_df, ignored_row_indices, ignored_reasons
    except Exception as e:
        print(f"CRITICAL ERROR during data cleaning helper: {e}"); traceback.print_exc()
        return None, original_transactions_df, ignored_row_indices, ignored_reasons # Propagate error

# --- IRR/MWR Calculation Functions ---
def calculate_npv(rate: float, dates: List[date], cash_flows: List[float]) -> float:
    if not isinstance(rate, (int, float)) or not np.isfinite(rate): return np.nan
    if len(dates) != len(cash_flows): raise ValueError("Dates and cash_flows must have the same length.")
    if not dates: return 0.0
    base = 1.0 + rate
    if base <= 1e-9: return np.nan # Avoid issues with rate = -1 or less
    start_date = dates[0]; npv = 0.0
    for i in range(len(cash_flows)):
        try:
            if not isinstance(dates[i], date) or not isinstance(start_date, date): return np.nan # Basic type check
            time_delta_years = (dates[i] - start_date).days / 365.0
            if not np.isfinite(time_delta_years) or (time_delta_years < -1e-9 and i > 0): return np.nan
            if not np.isfinite(cash_flows[i]): continue # Skip this flow
            if abs(base) < 1e-9 and time_delta_years != 0 : return np.nan
            if base >=0 or time_delta_years == int(time_delta_years):
                 denominator = base ** time_delta_years
            elif base < 0: return np.nan
            else: return np.nan # Should not happen, means time_delta_years is not integer and base is negative
            if not np.isfinite(denominator) or abs(denominator) < 1e-12: return np.nan # Avoid division by zero or infinity
            term_value = cash_flows[i] / denominator
            if not np.isfinite(term_value): return np.nan
            npv += term_value
        except (TypeError, OverflowError) as e: return np.nan
        except Exception as e: return np.nan # Catch any other unexpected calculation errors
    return npv if np.isfinite(npv) else np.nan

def calculate_irr(dates: List[date], cash_flows: List[float]) -> float:
    """
    Calculates Internal Rate of Return (IRR/MWR).
    Returns np.nan for invalid cases, including non-standard investment patterns
    (requires first non-zero flow to be negative and at least one later positive flow).
    """
    # 1. Basic Input Validation
    if len(dates) < 2 or len(cash_flows) < 2 or len(dates) != len(cash_flows):
        # print("DEBUG IRR: Fail - Length mismatch or < 2") # Optional Debug
        return np.nan
    if any(not isinstance(cf, (int, float)) or not np.isfinite(cf) for cf in cash_flows):
        # print("DEBUG IRR: Fail - Non-finite cash flows") # Optional Debug
        return np.nan
    # Check dates are valid and sorted
    try:
        # Ensure elements are date objects before comparison
        if not all(isinstance(d, date) for d in dates):
             raise TypeError("Not all elements in dates are date objects")
        for i in range(1, len(dates)):
            if dates[i] < dates[i-1]:
                raise ValueError("Dates are not sorted")
    except (TypeError, ValueError) as e:
        # print(f"DEBUG IRR: Fail - Date validation error: {e}") # Optional Debug
        return np.nan

    # 2. Cash Flow Pattern Validation (Stricter)
    first_non_zero_flow = None
    first_non_zero_idx = -1
    non_zero_cfs_list = [] # Also collect non-zero flows

    for idx, cf in enumerate(cash_flows):
        if abs(cf) > 1e-9:
            non_zero_cfs_list.append(cf)
            if first_non_zero_flow is None:
                first_non_zero_flow = cf
                first_non_zero_idx = idx

    if first_non_zero_flow is None: # All flows are zero or near-zero
        # print("DEBUG IRR: Fail - All flows are zero") # Optional Debug
        return np.nan

    # Stricter Check 1: First non-zero flow MUST be negative (typical investment)
    if first_non_zero_flow >= -1e-9:
        # print(f"DEBUG IRR: Fail - First non-zero flow is non-negative: {first_non_zero_flow} in {cash_flows}") # Optional Debug
        return np.nan

    # Stricter Check 2: Must have at least one positive flow overall
    has_positive_flow = any(cf > 1e-9 for cf in non_zero_cfs_list)
    if not has_positive_flow:
        # print(f"DEBUG IRR: Fail - No positive flows found: {cash_flows}") # Optional Debug
        return np.nan

    # Check 3 (Redundant but safe): Ensure not ALL flows are negative (covered by check 2)
    # all_negative = all(cf <= 1e-9 for cf in non_zero_cfs_list)
    # if all_negative:
    #     return np.nan

    # 3. Solver Logic (Keep previous robust version)
    irr_result = np.nan
    try:
        # Newton-Raphson with validation
        irr_result = optimize.newton(calculate_npv, x0=0.1, args=(dates, cash_flows), tol=1e-6, maxiter=100)
        if not np.isfinite(irr_result) or irr_result <= -1.0 or irr_result > 100.0: # Check range
             raise RuntimeError("Newton result out of reasonable range")
        npv_check = calculate_npv(irr_result, dates, cash_flows)
        if not np.isclose(npv_check, 0.0, atol=1e-4): # Check if it finds the root accurately
             raise RuntimeError("Newton result did not produce zero NPV")

    except (RuntimeError, OverflowError):
        # Brentq fallback
        try:
            lower_bound, upper_bound = -0.9999, 50.0 # Sensible bounds
            npv_low = calculate_npv(lower_bound, dates, cash_flows)
            npv_high = calculate_npv(upper_bound, dates, cash_flows)
            if pd.notna(npv_low) and pd.notna(npv_high) and npv_low * npv_high < 0: # Check sign change
                irr_result = optimize.brentq(calculate_npv, a=lower_bound, b=upper_bound, args=(dates, cash_flows), xtol=1e-6, rtol=1e-6, maxiter=100)
                # Final check on Brentq result
                if not np.isfinite(irr_result) or irr_result <= -1.0:
                    irr_result = np.nan
            else: # Bounds don't bracket
                irr_result = np.nan
        except (ValueError, RuntimeError, OverflowError, Exception):
             irr_result = np.nan # Brentq failed

    # 4. Final Validation and Return
    if not (isinstance(irr_result, (float, int)) and np.isfinite(irr_result) and irr_result > -1.0):
         # print(f"DEBUG IRR: Fail - Final result invalid: {irr_result}") # Optional Debug
         return np.nan

    return irr_result

# --- Cash Flow Helpers ---
# --- START OF REVISED get_cash_flows_for_symbol_account ---
def get_cash_flows_for_symbol_account(symbol: str, account: str, transactions: pd.DataFrame, final_market_value_local: float, end_date: date) -> Tuple[List[date], List[float]]:
    """Extracts LOCAL currency cash flows for symbol/account IRR."""
    # Assumes 'transactions' df ALREADY contains 'Local Currency' column.
    symbol_account_tx_filtered = transactions[(transactions['Symbol'] == symbol) & (transactions['Account'] == account)]
    if symbol_account_tx_filtered.empty: return [], []
    symbol_account_tx = symbol_account_tx_filtered.copy()
    dates_flows = defaultdict(float)
    symbol_account_tx.sort_values(by=['Date', 'original_index'], inplace=True, ascending=True)

    for _, row in symbol_account_tx.iterrows():
        tx_type = str(row.get('Type', '')).lower().strip()
        # --- Retrieve values ---
        qty_val = row.get('Quantity'); price_val = row.get('Price/Share'); commission_val = row.get('Commission'); total_amount_val = row.get('Total Amount')
        # --- Convert and handle potential NaNs ---
        qty = pd.to_numeric(qty_val, errors='coerce')
        price_local = pd.to_numeric(price_val, errors='coerce')

        # --- FIX: Handle NaN for scalar commission ---
        commission_local_raw = pd.to_numeric(commission_val, errors='coerce')
        commission_local = 0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        # --- END FIX ---

        total_amount_local = pd.to_numeric(total_amount_val, errors='coerce')
        tx_date = row['Date'].date(); cash_flow_local = 0.0; qty_abs = abs(qty) if pd.notna(qty) else 0.0

        # --- Calculations (unchanged logic, operates on local currency values) ---
        # ... (Calculations for buy, sell, short, dividend, fees, split remain the same) ...
        if tx_type == 'buy' or tx_type == 'deposit':
            if pd.notna(qty) and qty > 0 and pd.notna(price_local): cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == 'sell' or tx_type == 'withdrawal':
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0: cash_flow_local = (qty_abs * price_local) - commission_local
        elif tx_type == 'short sell' and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0: cash_flow_local = (qty_abs * price_local) - commission_local
        elif tx_type == 'buy to cover' and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0: cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == 'dividend':
             dividend_amount_local_cf = 0.0
             if pd.notna(total_amount_local): dividend_amount_local_cf = total_amount_local
             elif pd.notna(price_local): dividend_amount_local_cf = (qty_abs * price_local) if (pd.notna(qty) and qty_abs > 0) else price_local
             if pd.notna(dividend_amount_local_cf): cash_flow_local = dividend_amount_local_cf - commission_local
        elif tx_type == 'fees': cash_flow_local = -(abs(commission_local))
        elif tx_type in ['split', 'stock split']:
            cash_flow_local = 0.0;
            if commission_local != 0: cash_flow_local = -abs(commission_local)

        # --- Aggregation (unchanged) ---
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            try: flow_to_add = float(cash_flow_local); dates_flows[tx_date] += flow_to_add
            except (ValueError, TypeError): print(f"Warning IRR CF Gen ({symbol}/{account}): Could not convert cash_flow_local {cash_flow_local} to float. Skipping flow.")

    # --- Final sorting, adding final value, checks (unchanged) ---
    # ... (Rest of the function remains the same) ...
    sorted_dates = sorted(dates_flows.keys())
    if not sorted_dates and abs(final_market_value_local) < 1e-9 : return [],[]
    final_dates = list(sorted_dates); final_flows = [float(dates_flows[d]) for d in final_dates]
    final_market_value_local_abs = abs(final_market_value_local) if pd.notna(final_market_value_local) else 0.0
    if final_market_value_local_abs > 1e-9 and isinstance(end_date, date):
        # Ensure we only add final value if there are initial cash flows OR if the final value itself exists
        if not final_dates:
            # No prior cash flows, but a final value exists. Need a dummy start date?
            # Let's assume the first transaction date for the holding is the "start"
            first_tx_date_for_holding = symbol_account_tx['Date'].min().date() if not symbol_account_tx.empty else end_date
            if end_date >= first_tx_date_for_holding:
                 final_dates.append(end_date)
                 final_flows.append(final_market_value_local_abs)
            else: # End date is before the first transaction? Should not happen if holding exists.
                 return [], []
        elif end_date >= final_dates[-1]:
             if final_dates[-1] == end_date: final_flows[-1] += final_market_value_local_abs
             else: final_dates.append(end_date); final_flows.append(final_market_value_local_abs)
        # If end_date is before the last cash flow, the final value shouldn't be added here. calculate_irr handles time value.

    if len(final_dates) < 2: return [], [] # Need at least two points for IRR (e.g., initial investment + final value)
    non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
    if not non_zero_final_flows or all(cf >= -1e-9 for cf in non_zero_final_flows) or all(cf <= 1e-9 for cf in non_zero_final_flows): return [], []
    return final_dates, final_flows
# --- END OF REVISED get_cash_flows_for_symbol_account ---

# --- START OF REVISED get_cash_flows_for_mwr ---
def get_cash_flows_for_mwr(
    account_transactions: pd.DataFrame,
    final_account_market_value: float, # Already in target_currency
    end_date: date,
    target_currency: str,
    fx_rates: Optional[Dict[str, float]], # Expects standard 'FROM/TO' -> rate format
    display_currency: str # Used for warning msg only (REMOVED - fx_rates needed)
) -> Tuple[List[date], List[float]]:
    """Calculates MWR cash flows in target_currency using provided FX rates."""
    # Assumes 'account_transactions' df ALREADY contains 'Local Currency' column.
    if account_transactions.empty: return [], []
    acc_tx_copy = account_transactions.copy(); dates_flows = defaultdict(float)
    acc_tx_copy.sort_values(by=['Date', 'original_index'], inplace=True, ascending=True)

    for _, row in acc_tx_copy.iterrows():
        # ... (Get tx_type, symbol, qty, price, total_amount - unchanged) ...
        tx_type = str(row.get('Type', '')).lower().strip(); symbol = row['Symbol']
        qty = pd.to_numeric(row['Quantity'], errors='coerce'); price_local = pd.to_numeric(row['Price/Share'], errors='coerce'); commission_val = row.get('Commission'); total_amount_local = pd.to_numeric(row.get('Total Amount'), errors='coerce')

        # --- FIX: Handle NaN for scalar commission ---
        commission_local_raw = pd.to_numeric(commission_val, errors='coerce')
        commission_local = 0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        # --- END FIX ---

        tx_date = row['Date'].date()
        # --- Get Local Currency from the DataFrame ---
        local_currency = row['Local Currency']
        # -------------------------------------------
        cash_flow_local = 0.0; qty_abs = abs(qty) if pd.notna(qty) else 0.0

        # --- MWR Flow Logic (unchanged, calculates flow in local currency) ---
        # ... (Calculations for buy, sell, short, dividend, fees, split, cash remain the same) ...
        if symbol != CASH_SYMBOL_CSV:
            if tx_type == 'buy':
                if pd.notna(qty) and qty > 0 and pd.notna(price_local): cash_flow_local = -((qty_abs * price_local) + commission_local) # OUT (-)
            elif tx_type == 'sell':
                if pd.notna(price_local) and qty_abs > 0: cash_flow_local = (qty_abs * price_local) - commission_local # IN (+)
            elif tx_type == 'short sell' and symbol in SHORTABLE_SYMBOLS:
                 if pd.notna(price_local) and qty_abs > 0: cash_flow_local = (qty_abs * price_local) - commission_local # IN (+)
            elif tx_type == 'buy to cover' and symbol in SHORTABLE_SYMBOLS:
                 if pd.notna(price_local) and qty_abs > 0: cash_flow_local = -((qty_abs * price_local) + commission_local) # OUT (-)
            elif tx_type == 'dividend':
                 dividend_amount_local_cf = 0.0
                 if pd.notna(total_amount_local) and total_amount_local != 0: dividend_amount_local_cf = total_amount_local
                 elif pd.notna(price_local) and price_local != 0: dividend_amount_local_cf = (qty_abs * price_local) if qty_abs > 0 else price_local
                 cash_flow_local = dividend_amount_local_cf - commission_local # IN (+)
            elif tx_type == 'fees':
                 if pd.notna(commission_local): cash_flow_local = -(abs(commission_local)) # OUT (-)
            elif tx_type in ['split', 'stock split']:
                 cash_flow_local = 0.0
                 if pd.notna(commission_local) and commission_local != 0: cash_flow_local = -abs(commission_local) # OUT (-)
        elif symbol == CASH_SYMBOL_CSV:
            if tx_type == 'deposit' or tx_type == 'buy':
                if pd.notna(qty): cash_flow_local = abs(qty) # IN (+)
                cash_flow_local -= commission_local
            elif tx_type == 'withdrawal' or tx_type == 'sell':
                 if pd.notna(qty): cash_flow_local = -abs(qty) # OUT (-)
                 cash_flow_local -= commission_local
            elif tx_type == 'dividend':
                 dividend_amount_local_cf = 0.0
                 if pd.notna(total_amount_local) and total_amount_local != 0: dividend_amount_local_cf = total_amount_local
                 elif pd.notna(price_local) and price_local != 0: dividend_amount_local_cf = (qty_abs * price_local) if qty_abs > 0 else price_local
                 cash_flow_local = dividend_amount_local_cf - commission_local # IN (+)
            elif tx_type == 'fees':
                 if pd.notna(commission_local): cash_flow_local = -abs(commission_local) # OUT (-)

        # --- Convert to target currency ---
        cash_flow_target = cash_flow_local
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            if local_currency != target_currency:
                # --- Use get_conversion_rate helper ---
                rate = get_conversion_rate(local_currency, target_currency, fx_rates)
                # --------------------------------------
                if rate == 1.0 and local_currency != target_currency: # Rate lookup failed or was 1.0 incorrectly
                     # print(f"Warning: MWR calc cannot convert flow on {tx_date} from {local_currency} to {target_currency} (FX rate missing/invalid). Skipping flow.") # Reduced verbosity
                     cash_flow_target = 0.0 # Assign 0 instead of NaN to prevent downstream errors
                else:
                    cash_flow_target = cash_flow_local * rate

            if pd.notna(cash_flow_target) and abs(cash_flow_target) > 1e-9:
                dates_flows[tx_date] += cash_flow_target
            # else: Handle potential NaN after conversion if needed, currently skipped implicitly

    # --- Final sorting, adding final value, sign flip, checks (unchanged) ---
    # ... (Rest of the function remains the same) ...
    sorted_dates = sorted(dates_flows.keys())
    # If no cash flows generated and final MV is zero, return empty
    if not sorted_dates and abs(final_account_market_value) < 1e-9: return [], []

    final_dates = list(sorted_dates); final_flows = [-dates_flows[d] for d in final_dates] # <<< SIGN FLIP

    # --- Ensure final_account_market_value is treated as float ---
    final_market_value_target = float(final_account_market_value) if pd.notna(final_account_market_value) else 0.0
    final_market_value_abs = abs(final_market_value_target)

    if final_market_value_abs > 1e-9 and isinstance(end_date, date):
        # If no initial flows, add the first tx date (or end date) and the final value
        if not final_dates:
             first_tx_date_for_account = acc_tx_copy['Date'].min().date() if not acc_tx_copy.empty else end_date
             if end_date >= first_tx_date_for_account:
                 # Need a starting point for the MWR calc - usually a deposit or buy
                 # If only a final value exists, MWR is undefined. Let's return empty.
                 # Or perhaps add a zero flow at the start? Let's return empty for now.
                 return [], [] # Cannot calculate MWR with only a final value
             else:
                 return [], []
        elif end_date >= final_dates[-1]:
             if final_dates[-1] == end_date: final_flows[-1] += final_market_value_abs
             else: final_dates.append(end_date); final_flows.append(final_market_value_abs)
        # If end_date is before the last cash flow, final value isn't added here.

    if len(final_dates) < 2: return [], []
    non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
    if not non_zero_final_flows or all(cf >= -1e-9 for cf in non_zero_final_flows) or all(cf <= 1e-9 for cf in non_zero_final_flows): return [], []
    return final_dates, final_flows
# --- END OF REVISED get_cash_flows_for_mwr ---

# --- Current Price Fetching etc. ---
def fetch_index_quotes_yfinance(query_symbols: List[str] = DEFAULT_INDEX_QUERY_SYMBOLS) -> Dict[str, Optional[Dict[str, Any]]]:
    """ Fetches near real-time quotes for specified INDEX symbols using yfinance. """
    if not query_symbols: return {}
    yf_tickers_to_fetch = []; yf_ticker_to_query_symbol_map = {}
    for q_sym in query_symbols:
        yf_ticker = YFINANCE_INDEX_TICKER_MAP.get(q_sym)
        if yf_ticker:
            if yf_ticker not in yf_ticker_to_query_symbol_map: yf_tickers_to_fetch.append(yf_ticker)
            yf_ticker_to_query_symbol_map[yf_ticker] = q_sym
        else: print(f"Warn: No yfinance ticker mapping for index: {q_sym}")
    if not yf_tickers_to_fetch: return {q_sym: None for q_sym in query_symbols}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    # print(f"Fetching index quotes via yfinance for tickers: {', '.join(yf_tickers_to_fetch)}") # Reduced verbosity
    try:
        tickers_data = yf.Tickers(" ".join(yf_tickers_to_fetch))
        for yf_ticker, ticker_obj in tickers_data.tickers.items():
            original_query_symbol = yf_ticker_to_query_symbol_map.get(yf_ticker)
            if not original_query_symbol: continue
            ticker_info = getattr(ticker_obj, 'info', None)
            if ticker_info:
                try:
                    price_raw = ticker_info.get('regularMarketPrice', ticker_info.get('currentPrice'))
                    prev_close_raw = ticker_info.get('previousClose')
                    name = ticker_info.get('shortName', ticker_info.get('longName', original_query_symbol))
                    change_val_raw = ticker_info.get('regularMarketChange')
                    change_pct_val_raw = ticker_info.get('regularMarketChangePercent') # Raw value from yfinance

                    price = float(price_raw) if price_raw is not None else np.nan
                    prev_close = float(prev_close_raw) if prev_close_raw is not None else np.nan
                    change = float(change_val_raw) if change_val_raw is not None else np.nan

                    # --- FIX: Store raw percentage value, handle fallback without * 100 ---
                    # Get the percentage directly if available
                    changesPercentage = float(change_pct_val_raw) if change_pct_val_raw is not None else np.nan

                    # Calculate change if missing
                    if pd.isna(change) and pd.notna(price) and pd.notna(prev_close):
                        change = price - prev_close

                    # Fallback for percentage ONLY if primary fetch failed AND change/prev_close are valid
                    if pd.isna(changesPercentage) and pd.notna(change) and pd.notna(prev_close) and prev_close != 0:
                        # changesPercentage = (change / prev_close) * 100 # Scale to % <-- REMOVED * 100
                        changesPercentage = (change / prev_close) # Store as fraction/decimal
                    # --- END FIX ---

                    if pd.notna(price):
                        results[original_query_symbol] = {
                            'price': price,
                            'change': change if pd.notna(change) else np.nan,
                            'changesPercentage': changesPercentage if pd.notna(changesPercentage) else np.nan, # Use the potentially calculated value
                            'name': name,
                            'symbol': original_query_symbol,
                            'yf_ticker': yf_ticker
                        }
                    else:
                        results[original_query_symbol] = None; # print(f"Warn: Index {original_query_symbol} ({yf_ticker}) missing price.") # Reduced verbosity
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    results[original_query_symbol] = None; print(f"Warn: Error parsing index {original_query_symbol} ({yf_ticker}): {e}")
            else:
                results[original_query_symbol] = None; # print(f"Warn: No yfinance info for index: {yf_ticker}") # Reduced verbosity

    except Exception as e: print(f"Error fetching yfinance index data: {e}"); traceback.print_exc(); results = {q_sym: None for q_sym in query_symbols}
    for q_sym in query_symbols:
        if q_sym not in results: results[q_sym] = None
    return results

# In portfolio_logic.py

# Replace the ENTIRE get_cached_or_fetch_yfinance_data function with this:

def get_cached_or_fetch_yfinance_data(
    internal_stock_symbols: List[str],
    required_currencies: Set[str], # Currencies needed (incl. display_currency, default_currency, local currencies)
    cache_file: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    cache_duration_hours: int = YFINANCE_CACHE_DURATION_HOURS
) -> Tuple[Optional[Dict[str, Dict[str, Optional[float]]]], Optional[Dict[str, float]]]:
    """
    Gets stock/ETF price/change data and FX rates relative to USD using yfinance, leveraging a cache.
    Args:
        internal_stock_symbols: List of internal stock/ETF symbols.
        required_currencies: Set of currency codes (e.g., {'USD', 'EUR', 'THB'}) that need rates vs USD.
        cache_file: Path to the JSON cache file.
        cache_duration_hours: How long the cache is considered valid.
    Returns:
        Tuple containing:
        - Stock Data Dict: {internal_symbol: {'price': float, 'change': float, 'changesPercentage': float, 'previousClose': float}} or None on failure.
        - FX Rates vs USD Dict: {currency_code: rate_per_USD} (e.g., {'JPY': 143.5, 'EUR': 0.85}) or None on failure.
          Note: USD key will have value 1.0.
    """
    stock_data_internal: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    # --- MODIFIED: This dictionary now ONLY stores rates relative to USD ---
    fx_rates_vs_usd: Optional[Dict[str, float]] = {'USD': 1.0} # Initialize with USD base
    # --- END MODIFICATION ---

    cache_needs_update = True
    cached_data = {}
    now = datetime.now()

    # --- Cache Loading Logic ---
    if not cache_file:
        print(f"Warning: Invalid cache file path provided ('{cache_file}'). Cache read skipped.")
        cache_needs_update = True
    elif os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f: cached_data = json.load(f)
            cache_timestamp_str = cached_data.get('timestamp')
            cached_stocks = cached_data.get('stock_data_internal')
            # --- MODIFIED: Load fx_rates_vs_usd from cache ---
            cached_fx_vs_usd = cached_data.get('fx_rates_vs_usd')
            # --- END MODIFICATION ---

            if cache_timestamp_str and isinstance(cached_stocks, dict) and isinstance(cached_fx_vs_usd, dict):
                cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                cache_age = now - cache_timestamp
                if cache_age <= timedelta(hours=cache_duration_hours):
                    # Check if all required stocks and currencies are present
                    all_stocks_present = all(s in cached_stocks for s in internal_stock_symbols if s not in YFINANCE_EXCLUDED_SYMBOLS and s != CASH_SYMBOL_CSV)
                    stock_format_ok = all(isinstance(v, dict) and 'price' in v for v in cached_stocks.values()) if cached_stocks else True
                    # --- MODIFIED: Check required currencies in cached_fx_vs_usd ---
                    all_fx_present = required_currencies.issubset(cached_fx_vs_usd.keys())
                    # --- END MODIFICATION ---

                    if all_stocks_present and stock_format_ok and all_fx_present:
                        print(f"Yahoo Finance cache is valid (age: {cache_age}). Using cached data.")
                        stock_data_internal = cached_stocks
                        fx_rates_vs_usd = cached_fx_vs_usd
                        cache_needs_update = False
                    else:
                        missing_reason = []
                        if not all_stocks_present: missing_reason.append("stocks")
                        if not stock_format_ok: missing_reason.append("stock_format")
                        if not all_fx_present: missing_reason.append("currencies")
                        print(f"Yahoo Finance cache is recent but incomplete/invalid ({', '.join(missing_reason)}). Will fetch/update.")
                        # Start with cached data but mark for update
                        stock_data_internal = cached_stocks if stock_format_ok else {}
                        fx_rates_vs_usd = cached_fx_vs_usd if isinstance(cached_fx_vs_usd, dict) else {'USD': 1.0}
                        cache_needs_update = True
                else:
                    print(f"Yahoo Finance cache is outdated (age: {cache_age}). Will fetch fresh data.")
                    cache_needs_update = True
            else:
                print("Yahoo Finance cache is invalid or missing data. Will fetch fresh data.")
                cache_needs_update = True
        except Exception as e:
            print(f"Error reading Yahoo Finance cache file {cache_file}: {e}. Will fetch fresh data.")
            cache_needs_update = True
            # Reset to defaults on error
            stock_data_internal = {}
            fx_rates_vs_usd = {'USD': 1.0}
    else:
        print(f"Yahoo Finance cache file {cache_file} not found. Will fetch fresh data.")
        cache_needs_update = True

    # --- Data Fetching Logic ---
    if cache_needs_update:
        print("Fetching/Updating data from Yahoo Finance (Stocks & FX vs USD)...")
        # Initialize from cache if available, otherwise empty/default
        fetched_stocks = stock_data_internal if stock_data_internal is not None else {}
        # fx_rates_vs_usd is already initialized above (from cache or default {'USD': 1.0})

        # --- Determine symbols/tickers to fetch ---
        yf_tickers_to_fetch = set()
        internal_to_yf_map = {}
        yf_to_internal_map = {}
        missing_stock_symbols = []
        explicitly_excluded_count = 0

        # Stocks/ETFs
        for internal_sym in internal_stock_symbols:
            if internal_sym == CASH_SYMBOL_CSV: continue
            if internal_sym in YFINANCE_EXCLUDED_SYMBOLS:
                if internal_sym not in fetched_stocks:
                    explicitly_excluded_count += 1
                    fetched_stocks[internal_sym] = {'price': np.nan, 'change': np.nan, 'changesPercentage': np.nan, 'previousClose': np.nan}
                continue

            yf_ticker = SYMBOL_MAP_TO_YFINANCE.get(internal_sym, internal_sym.upper()) # Default to upper if not in map
            # --- MODIFICATION: Check if yf_ticker is potentially invalid BEFORE adding ---
            # Simple check: Avoid symbols with spaces or likely invalid chars if not mapped
            if yf_ticker and ' ' not in yf_ticker and ':' not in yf_ticker:
                should_fetch_stock = True
                if internal_sym in fetched_stocks:
                    cached_entry = fetched_stocks[internal_sym]
                    # Only skip fetch if price is valid in cache
                    if isinstance(cached_entry, dict) and pd.notna(cached_entry.get('price')) and cached_entry.get('price') > 1e-9:
                        should_fetch_stock = False

                if should_fetch_stock:
                    yf_tickers_to_fetch.add(yf_ticker)
                    internal_to_yf_map[internal_sym] = yf_ticker
                    yf_to_internal_map[yf_ticker] = internal_sym
            # --- END MODIFICATION ---
            else:
                missing_stock_symbols.append(internal_sym)
                if internal_sym not in fetched_stocks:
                    print(f"Warning: No valid Yahoo Finance ticker mapping for: {internal_sym}.")
                    fetched_stocks[internal_sym] = {'price': np.nan, 'change': np.nan, 'changesPercentage': np.nan, 'previousClose': np.nan}

        if explicitly_excluded_count > 0: print(f"Info: Explicitly excluded {explicitly_excluded_count} stock symbols.")

        # FX Rates vs USD
        fx_tickers_to_fetch = set()
        # --- MODIFIED: Determine which *USD based* rates are missing ---
        currencies_to_fetch_vs_usd = required_currencies - set(fx_rates_vs_usd.keys())
        for currency in currencies_to_fetch_vs_usd:
            if currency != 'USD' and currency and isinstance(currency, str): # Basic validation
                 # Fetch OTHERCURRENCY=X ticker which gives OTHERCURRENCY per 1 USD
                 fx_ticker = f"{currency}=X"
                 fx_tickers_to_fetch.add(fx_ticker)
                 yf_tickers_to_fetch.add(fx_ticker)
        # --- END MODIFICATION ---

        # --- Fetching using yfinance ---
        if not yf_tickers_to_fetch:
            print("No new Stock/FX tickers need fetching.")
        else:
            print(f"Fetching/Updating {len(yf_tickers_to_fetch)} tickers from Yahoo: {list(yf_tickers_to_fetch)}")
            fetch_success = True; all_fetched_data = {}
            try:
                yf_ticker_list = list(yf_tickers_to_fetch)
                fetch_batch_size = 50 # Adjust batch size if needed
                for i in range(0, len(yf_ticker_list), fetch_batch_size):
                    batch = yf_ticker_list[i:i+fetch_batch_size]
                    print(f"  Fetching batch {i//fetch_batch_size + 1} ({len(batch)} tickers)...")
                    try:
                        # Use yf.Tickers for batch fetching info
                        tickers_data = yf.Tickers(" ".join(batch))
                        for yf_ticker, ticker_obj in tickers_data.tickers.items():
                             ticker_info = getattr(ticker_obj, 'info', None)
                             price, change, pct_change, prev_close = np.nan, np.nan, np.nan, np.nan
                             if ticker_info:
                                 price_raw = ticker_info.get('regularMarketPrice', ticker_info.get('currentPrice')); change_raw = ticker_info.get('regularMarketChange'); pct_change_raw = ticker_info.get('regularMarketChangePercent'); prev_close_raw = ticker_info.get('previousClose')
                                 try: price = float(price_raw) if price_raw is not None else np.nan
                                 except (ValueError, TypeError): price = np.nan
                                 try: change = float(change_raw) if change_raw is not None else np.nan
                                 except (ValueError, TypeError): change = np.nan
                                 try: pct_change = float(pct_change_raw) if pct_change_raw is not None else np.nan # Already % from info? No, usually fraction.
                                 except (ValueError, TypeError): pct_change = np.nan
                                 try: prev_close = float(prev_close_raw) if prev_close_raw is not None else np.nan
                                 except (ValueError, TypeError): prev_close = np.nan
                                 # Ensure price is positive, otherwise treat as NaN
                                 if not (pd.notna(price) and price > 1e-9): price = np.nan
                                 # --- Convert pct_change fraction to percentage ---
                                 if pd.notna(pct_change): pct_change *= 100.0
                                 # --- END Convert ---
                             # Store whatever we got, even if incomplete
                             all_fetched_data[yf_ticker] = {'price': price, 'change': change, 'changesPercentage': pct_change, 'previousClose': prev_close }
                    except requests.exceptions.HTTPError as http_err: print(f"  HTTP ERROR fetching batch: {http_err}"); fetch_success = False;
                    except Exception as yf_err: print(f"  YFINANCE ERROR fetching batch info: {yf_err}"); fetch_success = False;
                    time.sleep(0.1) # Small delay
            except Exception as e: print(f"ERROR during Yahoo Finance fetch loop: {e}"); traceback.print_exc(); fetch_success = False;

            # --- Process fetched data ---
            # Stocks
            for yf_ticker, data_dict in all_fetched_data.items():
                internal_sym = yf_to_internal_map.get(yf_ticker)
                if internal_sym: fetched_stocks[internal_sym] = data_dict # Overwrite or add fetched data

            # Ensure all requested internal symbols have an entry (even if NaN)
            for sym in internal_stock_symbols:
                 if sym != CASH_SYMBOL_CSV and sym not in YFINANCE_EXCLUDED_SYMBOLS and sym not in fetched_stocks:
                     if sym not in missing_stock_symbols: print(f"Warning: Data for {sym} still not found after fetch.")
                     fetched_stocks[sym] = {'price': np.nan, 'change': np.nan, 'changesPercentage': np.nan, 'previousClose': np.nan}

            # FX Rates vs USD
            for currency in currencies_to_fetch_vs_usd: # Iterate over currencies we TRIED to fetch
                 if currency == 'USD': continue
                 fx_ticker = f"{currency}=X"; # Ticker we fetched (e.g., JPY=X)
                 fx_data_dict = all_fetched_data.get(fx_ticker);
                 price = fx_data_dict.get('price') if isinstance(fx_data_dict, dict) else np.nan
                 if pd.notna(price):
                      # Store the rate: Units of 'currency' per 1 USD
                      fx_rates_vs_usd[currency] = price
                 else:
                      print(f"Warning: Failed to fetch/update FX rate for {fx_ticker}. Previous value (if any) retained.")
                      # Ensure key exists even if fetch failed, maybe keep old value or set NaN?
                      if currency not in fx_rates_vs_usd:
                           fx_rates_vs_usd[currency] = np.nan

        # Assign final results after potential fetch/update
        stock_data_internal = fetched_stocks
        # fx_rates_vs_usd is updated in place

        # --- MODIFIED: Save Cache - ONLY store stock_data_internal and fx_rates_vs_usd ---
        if cache_needs_update: # Save only if we attempted an update
            if not cache_file:
                print(f"ERROR: Cache file path is invalid ('{cache_file}'). Cannot save cache.")
            else:
                print(f"Saving updated Yahoo Finance data (Stocks, FX vs USD) to cache: {cache_file}")
                # Ensure USD rate is present
                if 'USD' not in fx_rates_vs_usd: fx_rates_vs_usd['USD'] = 1.0
                # Prepare content
                content = {
                    'timestamp': now.isoformat(),
                    'stock_data_internal': stock_data_internal,
                    'fx_rates_vs_usd': fx_rates_vs_usd # Save the dictionary directly
                }
                try:
                    cache_dir = os.path.dirname(cache_file)
                    if cache_dir and not os.path.exists(cache_dir): os.makedirs(cache_dir)
                    with open(cache_file, 'w') as f:
                        # Use NaN-aware encoder
                        class NpEncoder(json.JSONEncoder):
                            def default(self, obj):
                                if isinstance(obj, np.integer): return int(obj)
                                if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None # Convert NaN to None for JSON
                                if isinstance(obj, np.ndarray): return obj.tolist()
                                return super(NpEncoder, self).default(obj)
                        json.dump(content, f, indent=4, cls=NpEncoder)
                except Exception as e:
                     print(f"Error writing Yahoo Finance cache ('{cache_file}'): {e}")
        # --- END MODIFICATION ---

    # --- Final Checks ---
    if not isinstance(stock_data_internal, dict): stock_data_internal = None
    if not isinstance(fx_rates_vs_usd, dict): fx_rates_vs_usd = None

    # Ensure USD is always in the returned FX dict if it's not None
    if isinstance(fx_rates_vs_usd, dict) and 'USD' not in fx_rates_vs_usd:
        fx_rates_vs_usd['USD'] = 1.0

    # Ensure all requested internal stock symbols have an entry, even if NaN
    if stock_data_internal is not None:
        for sym in internal_stock_symbols:
            if sym != CASH_SYMBOL_CSV and sym not in YFINANCE_EXCLUDED_SYMBOLS and sym not in stock_data_internal:
                 stock_data_internal[sym] = {'price': np.nan, 'change': np.nan, 'changesPercentage': np.nan, 'previousClose': np.nan}

    # --- MODIFIED: Return fx_rates_vs_usd ---
    return stock_data_internal, fx_rates_vs_usd
    # --- END MODIFICATION ---

# --- FINAL Version: Simple lookup assuming consistent dictionary ---
# Use this version of get_conversion_rate

def get_conversion_rate(from_curr: str, to_curr: str, fx_rates: Optional[Dict[str, float]]) -> float:
    """
    Gets CURRENT FX conversion rate (units of to_curr per 1 unit of from_curr).
    Calculates cross rates via USD using the provided fx_rates_vs_usd dictionary.
    Assumes fx_rates dictionary contains OTHER_CURRENCY per 1 USD.
    Returns 1.0 on failure.
    """
    if from_curr == to_curr:
        return 1.0
    if not isinstance(fx_rates, dict):
        # print(f"Warning: get_conversion_rate received invalid fx_rates type. Returning 1.0") # Optional
        return 1.0 # Fallback

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()

    # fx_rates now holds {CURRENCY: rate_per_USD} e.g., {'JPY': 143.3, 'THB': 33.5}

    # Get intermediate rates: Currency per 1 USD
    rate_A_per_USD = fx_rates.get(from_curr_upper) # e.g., THB per USD
    if from_curr_upper == 'USD': rate_A_per_USD = 1.0

    rate_B_per_USD = fx_rates.get(to_curr_upper)   # e.g., JPY per USD
    if to_curr_upper == 'USD': rate_B_per_USD = 1.0

    rate_B_per_A = np.nan # Initialize rate for B per A (TO / FROM)

    # Formula: TO / FROM = (TO / USD) / (FROM / USD)
    if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
        if abs(rate_A_per_USD) > 1e-9: # Check denominator (FROM/USD) is not zero
            try:
                rate_B_per_A = rate_B_per_USD / rate_A_per_USD
                # print(f"DEBUG get_conv_rate: {to_curr}/{from_curr} = {rate_B_per_USD} / {rate_A_per_USD} = {rate_B_per_A}") # DEBUG
            except (ZeroDivisionError, TypeError): pass # Keep NaN
        # else: Denominator is zero/invalid

    # Final check and fallback
    if pd.isna(rate_B_per_A):
        # print(f"Warning: Current FX rate lookup failed for {from_curr}->{to_curr}. Returning 1.0") # Optional Warning
        return 1.0
    else:
        return float(rate_B_per_A)
# --- Main Calculation Function (Current Portfolio Summary) ---
# (ensure it uses CASH_SYMBOL_CSV)
def calculate_portfolio_summary(
    transactions_csv_file: str,
    fmp_api_key: Optional[str] = None,
    display_currency: str = 'USD',
    show_closed_positions: bool = False,
    account_currency_map: Dict = {'SET': 'THB'}, # Keep default for signature
    default_currency: str = 'USD',             # Keep default for signature
    cache_file_path: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    include_accounts: Optional[List[str]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Dict[str, float]]], str]:
    """
    Calculates portfolio summary using Yahoo Finance for Stock/ETF price, change, prev close and FX rates.
    Filters calculations based on the `include_accounts` list.
    Uses `account_currency_map` and `default_currency` to determine local currencies.

    Args:
        transactions_csv_file (str): Path to the transactions CSV file.
        fmp_api_key (Optional[str]): FMP API key (currently unused for primary data).
        display_currency (str): The currency for displaying results.
        show_closed_positions (bool): Whether to include positions with zero quantity.
        account_currency_map (Dict): Map of account names to their local currency.
        default_currency (str): Default currency if account not in map.
        cache_file_path (str): Path for the Yahoo Finance current data cache.
        include_accounts (Optional[List[str]]): List of account names to include. If None or empty, all accounts are included.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Dict[str, float]]], str]:
            - Dictionary with overall summary metrics for the included accounts.
            - DataFrame with detailed holdings for the included accounts.
            - DataFrame with ignored transactions.
            - Dictionary with account-level metrics for the included accounts.
            - Status message string.
    """
    print(f"\n--- Starting Portfolio Calculation (Yahoo Finance) ---")
    filter_desc = "All Accounts"
    if include_accounts: filter_desc = f"Accounts: {', '.join(sorted(include_accounts))}"
    print(f"Parameters: CSV='{os.path.basename(transactions_csv_file)}', Currency='{display_currency}', ShowClosed={show_closed_positions}, Scope='{filter_desc}'")
    print(f"Currency Settings: Default='{default_currency}', Account Map='{account_currency_map}'") # Log map

    original_transactions_df: Optional[pd.DataFrame] = None
    all_transactions_df: Optional[pd.DataFrame] = None
    ignored_row_indices = set()
    ignored_reasons: Dict[int, str] = {}
    status_messages = []

    # --- 0. Initial Checks (unchanged) ---
    # ...

    # --- 1. Load & Clean ALL Transactions (Pass map/default) ---
    all_transactions_df, original_transactions_df, ignored_indices, ignored_reasons = _load_and_clean_transactions(
        transactions_csv_file,
        account_currency_map, # Pass map
        default_currency      # Pass default
    )
    if all_transactions_df is None:
        return None, None, None, None, f"Error: File not found or failed to load/clean: {transactions_csv_file}"
    if ignored_reasons: status_messages.append(f"Info: {len(ignored_reasons)} transactions ignored during load/clean.")

    # --- Get available accounts (unchanged) ---
    all_available_accounts_list = []
    if 'Account' in all_transactions_df.columns:
        all_available_accounts_list = sorted(all_transactions_df['Account'].unique().tolist())
        print(f"Logic: Found available accounts: {all_available_accounts_list}")

    # --- 1b. Filter Transactions based on include_accounts (unchanged) ---
    # ... (Filtering logic remains the same, uses all_transactions_df) ...
    transactions_df_filtered = pd.DataFrame()
    available_accounts = set(all_available_accounts_list)
    if not include_accounts:
        print("Info: No specific accounts provided for inclusion, using all available accounts.")
        transactions_df_filtered = all_transactions_df.copy()
        included_accounts_list = sorted(list(available_accounts))
    else:
        valid_include_accounts = [acc for acc in include_accounts if acc in available_accounts]
        if not valid_include_accounts:
            msg = "Warning: None of the specified accounts to include were found in the transactions. No data to process."
            print(msg); ignored_df_final = original_transactions_df.loc[sorted(list(ignored_indices))].copy() if ignored_indices and original_transactions_df is not None else pd.DataFrame(); # Add reason if possible...
            return {}, pd.DataFrame(), ignored_df_final, {}, msg
        print(f"Info: Filtering transactions FOR accounts: {', '.join(sorted(valid_include_accounts))}")
        transactions_df_filtered = all_transactions_df[all_transactions_df['Account'].isin(valid_include_accounts)].copy()
        included_accounts_list = sorted(valid_include_accounts)
    if transactions_df_filtered.empty:
        msg = f"Warning: No transactions remain after filtering for accounts: {', '.join(sorted(include_accounts))}"
        print(msg); ignored_df_final = original_transactions_df.loc[sorted(list(ignored_indices))].copy() if ignored_indices and original_transactions_df is not None else pd.DataFrame(); # Add reason if possible...
        return {}, pd.DataFrame(), ignored_df_final, {}, msg

    # --- Use transactions_df_filtered from here onwards ---
    transactions_df = transactions_df_filtered # Assign to the variable name used in subsequent steps

    # --- 3. Process Transactions (using filtered df) ---
    print("Processing filtered transactions...")
    holdings: Dict[Tuple[str, str], Dict] = {}
    overall_realized_gains_local: Dict[str, float] = defaultdict(float)
    overall_dividends_local: Dict[str, float] = defaultdict(float)
    overall_commissions_local: Dict[str, float] = defaultdict(float)

    # --- Loop over transactions_df (filtered) ---
    # Logic inside this loop is unchanged, it inherently uses the 'Local Currency' column added earlier.
    # ... (existing logic for buy, sell, dividend, split, short etc.) ...
    for index, row in transactions_df.iterrows():
        original_index = row['original_index']; symbol = row['Symbol']
        if symbol == CASH_SYMBOL_CSV: continue
        account, tx_type = row['Account'], row['Type']
        qty, price_local, total_amount_local = row['Quantity'], row['Price/Share'], row['Total Amount']
        commission_local, split_ratio = row['Commission'], row['Split Ratio']
        # --- Use Local Currency from DataFrame ---
        local_currency, tx_date = row['Local Currency'], row['Date'].date()
        # ---------------------------------------
        holding_key = (symbol, account)
        if holding_key not in holdings: holdings[holding_key] = { 'qty': 0.0, 'total_cost_local': 0.0, 'realized_gain_local': 0.0, 'dividends_local': 0.0, 'commissions_local': 0.0, 'local_currency': local_currency, 'short_proceeds_local': 0.0, 'short_original_qty': 0.0, 'total_cost_invested_local': 0.0, 'cumulative_investment_local': 0.0 }
        elif holdings[holding_key]['local_currency'] != local_currency:
             msg=f"Currency mismatch for {symbol}/{account}"; print(f"CRITICAL WARN: {msg} row {original_index}. Skip."); ignored_reasons[original_index]=msg; ignored_indices.add(original_index); continue
        # --- Rest of holding processing logic (buy/sell/short/div/split/fee) is unchanged ---
        try:
            holding = holdings[holding_key]; commission_for_overall = commission_local
            if symbol in SHORTABLE_SYMBOLS and tx_type in ['short sell', 'buy to cover']:
                qty_abs = abs(qty);
                if tx_type == 'short sell':
                    if qty_abs <= 1e-9: raise ValueError("Short Sell qty must be > 0")
                    proceeds = (qty_abs * price_local) - commission_local; holding['qty'] -= qty_abs; holding['short_proceeds_local'] += proceeds; holding['short_original_qty'] += qty_abs; holding['commissions_local'] += commission_local
                elif tx_type == 'buy to cover':
                    if qty_abs <= 1e-9: raise ValueError("Buy Cover qty must be > 0")
                    qty_currently_short = abs(holding['qty']) if holding['qty'] < -1e-9 else 0.0;
                    if qty_currently_short < 1e-9: raise ValueError(f"Not currently short {symbol}/{account} to cover.")
                    qty_covered = min(qty_abs, qty_currently_short); cost = (qty_covered * price_local) + commission_local
                    if holding['short_original_qty'] <= 1e-9: raise ZeroDivisionError(f"Short original qty is zero/neg for {symbol}/{account}")
                    avg_proceeds_per_share = holding['short_proceeds_local'] / holding['short_original_qty']; proceeds_attributed = qty_covered * avg_proceeds_per_share; gain = proceeds_attributed - cost
                    holding['qty'] += qty_covered; holding['short_proceeds_local'] -= proceeds_attributed; holding['short_original_qty'] -= qty_covered
                    holding['commissions_local'] += commission_local; holding['realized_gain_local'] += gain; overall_realized_gains_local[local_currency] += gain; holding['total_cost_invested_local'] += cost
                    if abs(holding['short_original_qty']) < 1e-9: holding['short_proceeds_local'] = 0.0; holding['short_original_qty'] = 0.0
                    if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0; holding['total_cost_local'] = 0.0
                    holding['cumulative_investment_local'] += cost
                continue
            if tx_type == 'buy' or tx_type == 'deposit':
                qty_abs = abs(qty);
                if qty_abs <= 1e-9: raise ValueError("Buy/Deposit qty must be > 0")
                cost = (qty_abs * price_local) + commission_local; holding['qty'] += qty_abs; holding['total_cost_local'] += cost; holding['commissions_local'] += commission_local; holding['total_cost_invested_local'] += cost; holding['cumulative_investment_local'] += cost
            elif tx_type == 'sell' or tx_type == 'withdrawal':
                qty_abs = abs(qty); held_qty = holding['qty'];
                if held_qty <= 1e-9: msg=f"Sell attempt {symbol}/{account} w/ non-positive long qty ({held_qty:.4f})"; print(f"Warn: {msg} row {original_index}. Skip."); ignored_reasons[original_index]=msg; ignored_indices.add(original_index); continue
                if qty_abs <= 1e-9: raise ValueError("Sell/Withdrawal qty must be > 0")
                qty_sold = min(qty_abs, held_qty); cost_sold = 0.0
                if held_qty > 1e-9 and abs(holding['total_cost_local']) > 1e-9: cost_sold = qty_sold * (holding['total_cost_local'] / held_qty)
                proceeds = (qty_sold * price_local) - commission_local; gain = proceeds - cost_sold
                holding['qty'] -= qty_sold; holding['total_cost_local'] -= cost_sold; holding['commissions_local'] += commission_local; holding['realized_gain_local'] += gain; overall_realized_gains_local[local_currency] += gain; holding['total_cost_invested_local'] -= cost_sold
                if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0; holding['total_cost_local'] = 0.0
            elif tx_type == 'dividend':
                div_amt = 0.0; qty_abs = abs(qty) if pd.notna(qty) else 0
                if pd.notna(total_amount_local) and total_amount_local != 0: div_amt = total_amount_local
                elif pd.notna(price_local) and price_local != 0: div_amt = (qty_abs * price_local) if qty_abs > 0 else price_local
                div_effect = abs(div_amt) if (holding['qty'] >= -1e-9 or symbol not in SHORTABLE_SYMBOLS) else -abs(div_amt)
                holding['dividends_local'] += div_effect; overall_dividends_local[local_currency] += div_effect; holding['commissions_local'] += commission_local
            elif tx_type == 'fees':
                 fee_cost = abs(commission_local); holding['commissions_local'] += fee_cost; holding['total_cost_invested_local'] += fee_cost; holding['cumulative_investment_local'] += fee_cost
            elif tx_type in ['split', 'stock split']:
                if pd.isna(split_ratio) or split_ratio <= 0: raise ValueError(f"Invalid split ratio: {split_ratio}")
                old_qty = holding['qty']
                if abs(old_qty) >= 1e-9:
                    holding['qty'] *= split_ratio
                    if old_qty < -1e-9 and symbol in SHORTABLE_SYMBOLS: holding['short_original_qty'] *= split_ratio
                    if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0
                    if abs(holding['short_original_qty']) < 1e-9: holding['short_original_qty'] = 0.0
                holding['commissions_local'] += commission_local; holding['total_cost_invested_local'] += commission_local; holding['cumulative_investment_local'] += commission_local
            else: msg=f"Unhandled stock tx type '{tx_type}'"; print(f"Warn: {msg} row {original_index}. Skip."); ignored_reasons[original_index]=msg; ignored_indices.add(original_index); commission_for_overall = 0.0; continue
            if commission_for_overall != 0: overall_commissions_local[local_currency] += abs(commission_for_overall)
        except (ValueError, TypeError, ZeroDivisionError, KeyError, Exception) as e:
            error_msg = f"Processing Error: {e}"; print(f"ERROR processing row {original_index} ({symbol}, {tx_type}): {e}. Skipping row."); traceback.print_exc(); ignored_reasons[original_index] = error_msg; ignored_indices.add(original_index); continue

    # --- 4. Calculate $CASH Balances (using filtered df) ---
    # Logic inside this block is unchanged, it inherently uses the 'Local Currency' column added earlier.
    # ... (existing logic for cash summary calculation) ...
    cash_summary: Dict[str, Dict] = {}
    try:
        cash_transactions = transactions_df[transactions_df['Symbol'] == CASH_SYMBOL_CSV].copy()
        if not cash_transactions.empty:
            def get_signed_quantity_cash(row): type_lower = row['Type']; qty = row['Quantity']; return 0.0 if pd.isna(qty) else (abs(qty) if type_lower in ['buy', 'deposit'] else (-abs(qty) if type_lower in ['sell', 'withdrawal'] else 0.0))
            cash_transactions['SignedQuantity'] = cash_transactions.apply(get_signed_quantity_cash, axis=1)
            cash_qty_agg = cash_transactions.groupby('Account')['SignedQuantity'].sum()
            cash_comm_agg = cash_transactions.groupby('Account')['Commission'].sum()
            cash_dividends_tx = cash_transactions[cash_transactions['Type'] == 'dividend'].copy()
            cash_div_agg = pd.Series(dtype=float)
            if not cash_dividends_tx.empty:
                 cash_dividends_tx['DividendAmount'] = cash_dividends_tx.apply(lambda r: r['Total Amount'] if pd.notna(r['Total Amount']) else ((abs(r['Quantity']) * r['Price/Share']) if pd.notna(r['Quantity']) and r['Quantity'] != 0 and pd.notna(r['Price/Share']) else (r['Price/Share'] if pd.notna(r['Price/Share']) else 0.0)), axis=1).fillna(0.0)
                 cash_dividends_tx['NetDividend'] = cash_dividends_tx['DividendAmount'] - cash_dividends_tx['Commission']
                 cash_div_agg = cash_dividends_tx.groupby('Account')['NetDividend'].sum()
            # --- Use Local Currency from DataFrame ---
            cash_currency_map = cash_transactions.groupby('Account')['Local Currency'].first()
            # ---------------------------------------
            all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index).union(cash_comm_agg.index).union(cash_div_agg.index)
            for acc in all_cash_accounts:
                acc_currency = cash_currency_map.get(acc, default_currency); acc_balance = cash_qty_agg.get(acc, 0.0); acc_commissions = cash_comm_agg.get(acc, 0.0); acc_dividends_only = cash_div_agg.get(acc, 0.0)
                cash_summary[acc] = { 'qty': acc_balance, 'realized': 0.0, 'dividends': acc_dividends_only, 'commissions': acc_commissions, 'currency': acc_currency } # Store currency
        else: print("Info: No $CASH transactions found in the filtered set.")
    except Exception as e: print(f"ERROR calculating $CASH balances separately: {e}"); traceback.print_exc(); cash_summary = {}

    # --- 5. Fetch Current Data (Prices, Change, FX Rates) via Yahoo (unchanged) ---
    # This determines required currencies based on the holdings derived from filtered transactions.
    # ... (Logic for determining symbols, required currencies, and calling get_cached_or_fetch_yfinance_data remains the same) ...
    print("Determining required data and fetching from Yahoo Finance...")
    report_date = datetime.now().date()
    all_stock_symbols_internal = list(set(key[0] for key in holdings.keys() if key[0] != CASH_SYMBOL_CSV))
    # Determine required currencies from BOTH holdings and cash summary
    required_currencies: Set[str] = set([display_currency, default_currency])
    for data in holdings.values(): required_currencies.add(data.get('local_currency', default_currency))
    for data in cash_summary.values(): required_currencies.add(data.get('currency', default_currency)) # Use 'currency' key for cash

    current_stock_data_internal, current_fx_rates_vs_usd = get_cached_or_fetch_yfinance_data(
        internal_stock_symbols=all_stock_symbols_internal,
        required_currencies=required_currencies,
        cache_file=cache_file_path
    )
    # And update the subsequent check:
    if current_stock_data_internal is None or current_fx_rates_vs_usd is None: # Check the new variable
        print("FATAL: Failed to fetch critical Stock/FX data from Yahoo Finance and/or cache. Cannot proceed.")
        ignored_df = original_transactions_df.loc[sorted(list(ignored_indices))].copy() if ignored_indices and original_transactions_df is not None else pd.DataFrame(); # Add reason if possible...
        return None, None, ignored_df, None, "Error: Price/FX/Change fetch failed via Yahoo Finance."


    # --- 6. Calculate Final Summary in DISPLAY_CURRENCY (unchanged logic) ---
    # This loop inherently uses the 'local_currency' determined earlier for each holding/cash position.
    # The call to get_conversion_rate uses this local currency.
    # ... (Calculation logic for market value, gains, IRR etc. remains the same) ...
    print(f"Calculating final portfolio summary in {display_currency}...")
    portfolio_summary = [] # List of dictionaries, one per holding row
    account_market_values_local: Dict[str, float] = defaultdict(float) # Sum of MV per account in local currency
    account_local_currency_map: Dict[str, str] = {} # Map account to its local currency

    for holding_key, data in holdings.items():
        # ... (Get symbol, account, qty, gains, etc.) ...
        symbol, account = holding_key; current_qty = data['qty']; realized_gain_local = data['realized_gain_local']; dividends_local = data['dividends_local']; commissions_local = data['commissions_local']; local_currency = data['local_currency']; current_total_cost_local = data['total_cost_local']; short_proceeds_local = data['short_proceeds_local']; short_original_qty = data['short_original_qty']; total_cost_invested_local = data['total_cost_invested_local']; cumulative_investment_local = data['cumulative_investment_local']
        account_local_currency_map[account] = local_currency
        # ... (Get current price, day change using Yahoo data) ...
        stock_data = current_stock_data_internal.get(symbol, {}); current_price_local = stock_data.get('price', np.nan); day_change_local = stock_data.get('change', np.nan); day_change_pct = stock_data.get('changesPercentage', np.nan); prev_close_local = stock_data.get('previousClose', np.nan)
        # ... (Price source determination and fallback logic) ...
        price_source = "Unknown"; is_yahoo_price_valid = pd.notna(current_price_local) and current_price_local > 0; is_excluded = symbol in YFINANCE_EXCLUDED_SYMBOLS
        if is_excluded: price_source = "Excluded - Fallback"; current_price_local = 0.0; day_change_local, day_change_pct, prev_close_local = np.nan, np.nan, np.nan; is_yahoo_price_valid = False
        elif is_yahoo_price_valid: price_source = "Yahoo API/Cache"
        else: price_source = "Yahoo Invalid - Fallback"; current_price_local = 0.0; is_yahoo_price_valid = False
        if not is_yahoo_price_valid: # Fallback
            try:
                 symbol_account_tx = transactions_df[(transactions_df['Symbol'] == symbol) & (transactions_df['Account'] == account) & (transactions_df['Price/Share'].notna()) & (transactions_df['Price/Share'] > 0)]
                 if not symbol_account_tx.empty:
                     last_tx_row = symbol_account_tx.iloc[-1]; last_tx_price = float(last_tx_row['Price/Share'])
                     if pd.notna(last_tx_price) and last_tx_price > 0: current_price_local = last_tx_price; price_source = "Last TX Fallback" + (" (Excluded)" if is_excluded else "")
                     else: price_source = "Fallback Zero (Bad TX)" + (" (Excluded)" if is_excluded else "")
                 else: price_source = "Fallback Zero (No TX)" + (" (Excluded)" if is_excluded else "")
            except Exception as e_fallback: price_source = "Fallback Zero (Error)" + (" (Excluded)" if is_excluded else ""); current_price_local = 0.0
            try: current_price_local = float(current_price_local)
            except (ValueError, TypeError): current_price_local = 0.0

        # --- Use get_conversion_rate helper ---
        fx_rate = get_conversion_rate(local_currency, display_currency, current_fx_rates_vs_usd)
        # --- Continue with calculations using fx_rate ---
        market_value_local = current_qty * current_price_local if pd.notna(current_price_local) else 0.0; market_value_display = market_value_local * fx_rate; account_market_values_local[account] += market_value_local; day_change_value_local = 0.0
        if price_source == "Yahoo API/Cache" and pd.notna(day_change_local): day_change_value_local = current_qty * day_change_local
        day_change_value_display = day_change_value_local * fx_rate; current_price_display = current_price_local * fx_rate if pd.notna(current_price_local) else np.nan; cost_basis_display, avg_cost_price_display, unrealized_gain_display, unrealized_gain_pct = 0.0, np.nan, 0.0, np.nan
        is_long, is_short = current_qty > 1e-9, current_qty < -1e-9
        if is_long:
             cost_basis_display = max(0, current_total_cost_local * fx_rate); avg_cost_price_display = (cost_basis_display / current_qty) if abs(current_qty) > 1e-9 else np.nan; unrealized_gain_display = market_value_display - cost_basis_display
             if abs(cost_basis_display) > 1e-9: unrealized_gain_pct = (unrealized_gain_display / cost_basis_display) * 100.0
             else: unrealized_gain_pct = np.inf if market_value_display > 1e-9 else (-np.inf if market_value_display < -1e-9 else 0.0)
        elif is_short:
             avg_cost_price_display = np.nan; cost_basis_display = 0.0; short_proceeds_display = short_proceeds_local * fx_rate; current_cost_to_cover_display = abs(market_value_display); unrealized_gain_display = short_proceeds_display - current_cost_to_cover_display
             if abs(short_proceeds_display) > 1e-9: unrealized_gain_pct = (unrealized_gain_display / short_proceeds_display) * 100.0
             else: unrealized_gain_pct = -np.inf if current_cost_to_cover_display > 1e-9 else 0.0
        realized_gain_display = realized_gain_local * fx_rate; dividends_display = dividends_local * fx_rate; commissions_display = commissions_local * fx_rate; unrealized_gain_component = unrealized_gain_display if pd.notna(unrealized_gain_display) else 0.0; total_gain_display = realized_gain_display + unrealized_gain_component + dividends_display - commissions_display; total_cost_invested_display = total_cost_invested_local * fx_rate; cumulative_investment_display = cumulative_investment_local * fx_rate
        total_return_pct = np.nan; denominator_for_pct = cumulative_investment_display
        if abs(denominator_for_pct) > 1e-9: total_return_pct = (total_gain_display / denominator_for_pct) * 100.0
        elif abs(total_gain_display) <= 1e-9: total_return_pct = 0.0

        # --- Calculate IRR ---
        market_value_local_for_irr = abs(market_value_local) if abs(current_qty) > 1e-9 else 0.0
        stock_irr = np.nan
        # --- V ADD IRR DEBUGGING ---
        # --- Set specific symbols you want to debug here ---
        # --- e.g., symbols you KNOW should have a valid IRR ---
        symbols_to_debug_irr = {'AAPL', 'MSFT', 'SPY', 'QQQ'} # Modify this set
        print_irr_debug = symbol in symbols_to_debug_irr
        if print_irr_debug:
             print(f"\n--- IRR Debug START for {symbol}/{account} ---")
             print(f"  Target Date (report_date): {report_date}")
             print(f"  Current Qty: {current_qty:.4f}")
             print(f"  Local Price: {current_price_local:.4f}")
             print(f"  Market Value (Local): {market_value_local:.4f}")
             print(f"  Market Value for IRR (Local): {market_value_local_for_irr:.4f}")
             print(f"  Calling get_cash_flows_for_symbol_account...")
        # --- ^ ADD IRR DEBUGGING ---
        try:
            cf_dates, cf_values = get_cash_flows_for_symbol_account(
                symbol, account, transactions_df, market_value_local_for_irr, report_date
            )
            # --- V ADD IRR DEBUGGING ---
            if print_irr_debug:
                 print(f"  Cash Flows Returned ({len(cf_dates)} dates):")
                 if cf_dates:
                     for d, v in zip(cf_dates, cf_values):
                         print(f"    {d}: {v:,.2f}")
                 else:
                     print("    No valid cash flows generated by get_cash_flows_for_symbol_account.")
            # --- ^ ADD IRR DEBUGGING ---

            if cf_dates and cf_values:
                # --- V ADD IRR DEBUGGING ---
                if print_irr_debug: print(f"  Calling calculate_irr...")
                # --- ^ ADD IRR DEBUGGING ---
                stock_irr = calculate_irr(cf_dates, cf_values)
                # --- V ADD IRR DEBUGGING ---
                if print_irr_debug: print(f"  IRR calculated by calculate_irr: {stock_irr}")
                # --- ^ ADD IRR DEBUGGING ---
            # --- V ADD IRR DEBUGGING ---
            # else: # Optional: Print why calculate_irr was skipped
            #    if print_irr_debug: print(f"  Skipping calculate_irr call (cf_dates or cf_values empty/invalid).")
            # --- ^ ADD IRR DEBUGGING ---
        except Exception as e_irr:
            # --- V ADD IRR DEBUGGING ---
            if print_irr_debug: print(f"  ERROR during IRR calculation process: {e_irr}")
            # --- ^ ADD IRR DEBUGGING ---
            stock_irr = np.nan # Suppress detailed print in production

        irr_value_to_store = stock_irr * 100.0 if pd.notna(stock_irr) else np.nan
        # --- V ADD IRR DEBUGGING ---
        if print_irr_debug:
             print(f"  Value being stored for 'IRR (%)': {irr_value_to_store}")
             print(f"--- IRR Debug END for {symbol}/{account} ---\n")
        # --- ^ ADD IRR DEBUGGING ---
        portfolio_summary.append({ 'Account': account, 'Symbol': symbol, 'Quantity': current_qty, f'Avg Cost ({display_currency})': avg_cost_price_display, f'Price ({display_currency})': current_price_display, f'Cost Basis ({display_currency})': cost_basis_display, f'Market Value ({display_currency})': market_value_display, f'Day Change ({display_currency})': day_change_value_display if pd.notna(day_change_value_display) else np.nan, 'Day Change %': day_change_pct, f'Unreal. Gain ({display_currency})': unrealized_gain_display, 'Unreal. Gain %': unrealized_gain_pct, f'Realized Gain ({display_currency})': realized_gain_display, f'Dividends ({display_currency})': dividends_display, f'Commissions ({display_currency})': commissions_display, f'Total Gain ({display_currency})': total_gain_display, f'Total Cost Invested ({display_currency})': total_cost_invested_display, 'Total Return %': total_return_pct, f'Cumulative Investment ({display_currency})': cumulative_investment_display, 'IRR (%)': stock_irr * 100.0 if pd.notna(stock_irr) else np.nan, 'Local Currency': local_currency, 'Price Source': price_source })

    # --- Process CASH balances (using local_currency from cash_summary) ---
    if cash_summary:
        for account, cash_data in cash_summary.items():
            # ... (Get symbol, qty, gains etc.) ...
            symbol = CASH_SYMBOL_CSV; current_qty = cash_data.get('qty', 0.0); local_currency = cash_data.get('currency', default_currency); realized_gain_local = cash_data.get('realized', 0.0); dividends_local = cash_data.get('dividends', 0.0); commissions_local = cash_data.get('commissions', 0.0)
            account_market_values_local[account] += current_qty; account_local_currency_map[account] = local_currency
            # --- Use get_conversion_rate helper ---
            fx_rate = get_conversion_rate(local_currency, display_currency, current_fx_rates_vs_usd)
            # --- Continue with calculations using fx_rate ---
            current_price_local = 1.0; current_price_display = current_price_local * fx_rate; market_value_display = current_qty * current_price_display; cost_basis_display = market_value_display; avg_cost_price_display = current_price_display; realized_gain_display = realized_gain_local * fx_rate; dividends_display = dividends_local * fx_rate; commissions_display = commissions_local * fx_rate; total_gain_display = realized_gain_display + 0.0 + dividends_display - commissions_display; cash_cumulative_investment_display = max(0.0, market_value_display); total_return_pct_cash = np.nan; cash_denominator_for_pct = cash_cumulative_investment_display
            if abs(cash_denominator_for_pct) > 1e-9: total_return_pct_cash = (total_gain_display / cash_denominator_for_pct) * 100.0
            elif abs(total_gain_display) <= 1e-9: total_return_pct_cash = 0.0
            portfolio_summary.append({ 'Account': account, 'Symbol': symbol, 'Quantity': current_qty, f'Avg Cost ({display_currency})': avg_cost_price_display, f'Price ({display_currency})': current_price_display, f'Cost Basis ({display_currency})': cost_basis_display, f'Market Value ({display_currency})': market_value_display, f'Day Change ({display_currency})': 0.0, 'Day Change %': 0.0, f'Unreal. Gain ({display_currency})': 0.0, 'Unreal. Gain %': 0.0, f'Realized Gain ({display_currency})': realized_gain_display, f'Dividends ({display_currency})': dividends_display, f'Commissions ({display_currency})': commissions_display, f'Total Gain ({display_currency})': total_gain_display, f'Total Cost Invested ({display_currency})': cost_basis_display, f'Cumulative Investment ({display_currency})': cash_cumulative_investment_display, 'Total Return %': total_return_pct_cash, 'IRR (%)': np.nan, 'Local Currency': local_currency, 'Price Source': 'N/A (Cash)' })

    # --- 7. Create Final DataFrame & Calculate Account/Overall Metrics (unchanged logic) ---
    # ... (The MWR calculation inside this section calls get_cash_flows_for_mwr, which now correctly uses local currency and converts) ...
    summary_df = pd.DataFrame()
    overall_summary_metrics = {}
    overall_portfolio_mwr = np.nan
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: { 'mwr': np.nan, 'total_return_pct': np.nan, 'total_market_value_display': 0.0, 'total_realized_gain_display': 0.0, 'total_unrealized_gain_display': 0.0, 'total_dividends_display': 0.0, 'total_commissions_display': 0.0, 'total_gain_display': 0.0, 'total_cash_display': 0.0, 'total_cost_invested_display': 0.0, 'total_day_change_display': 0.0, 'total_day_change_percent': np.nan })
    if not portfolio_summary:
        # ... (Handle empty summary) ...
        print("Warning: Portfolio summary list is empty after processing filtered holdings and cash.")
        overall_summary_metrics = { "market_value": 0.0, "cost_basis_held": 0.0, "unrealized_gain": 0.0, "realized_gain": 0.0, "dividends": 0.0, "commissions": 0.0, "total_gain": 0.0, "total_cost_invested":0.0, "portfolio_mwr": np.nan, "report_date": report_date.strftime('%Y-%m-%d'), "display_currency": display_currency, "day_change_display": 0.0, "day_change_percent": np.nan }
        ignored_df_final = original_transactions_df.loc[sorted(list(ignored_indices))].copy() if ignored_indices and original_transactions_df is not None else pd.DataFrame(); # Add reason...
        return overall_summary_metrics, pd.DataFrame(), ignored_df_final, {}, "Warning: No holdings data generated for selected accounts."
    else:
        full_summary_df = pd.DataFrame(portfolio_summary)
        # ... (Numeric conversion) ...
        money_cols_display = [c for c in full_summary_df.columns if f'({display_currency})' in c]; percent_cols = ['Unreal. Gain %', 'Total Return %', 'IRR (%)', 'Day Change %']; numeric_cols_to_convert = ['Quantity'] + money_cols_display + percent_cols
        if f'Cumulative Investment ({display_currency})' not in money_cols_display: money_cols_display.append(f'Cumulative Investment ({display_currency})')
        for col in numeric_cols_to_convert:
            if col in full_summary_df.columns: full_summary_df[col] = pd.to_numeric(full_summary_df[col], errors='coerce')
        full_summary_df.sort_values(by=['Account', f'Market Value ({display_currency})'], ascending=[True, False], na_position='last', inplace=True)
        print("Calculating Account-Level Metrics (MWR, Totals, Day Change) for included accounts...")
        unique_accounts_in_summary = full_summary_df['Account'].unique()
        for account in unique_accounts_in_summary:
            # ... (Account MWR calc setup) ...
            account_full_df = full_summary_df[full_summary_df['Account'] == account]
            account_tx = transactions_df[transactions_df['Account'] == account]
            account_currency = account_local_currency_map.get(account, default_currency)
            metrics_entry = account_level_metrics[account]
            account_mv_local = account_market_values_local.get(account, 0.0)
            account_mv_display = metrics_entry['total_market_value_display'] = pd.to_numeric(account_full_df[f'Market Value ({display_currency})'], errors='coerce').fillna(0.0).sum() # Sum MV display for MWR final value
            mwr = np.nan
            try:
                # Pass the account_mv_display (already converted)
                cf_dates_mwr, cf_values_mwr = get_cash_flows_for_mwr( account_tx, account_mv_display, report_date, display_currency, current_fx_rates_vs_usd, display_currency ) # Target currency is display_currency
                if cf_dates_mwr and cf_values_mwr: mwr = calculate_irr(cf_dates_mwr, cf_values_mwr)
                metrics_entry['mwr'] = mwr * 100.0 if pd.notna(mwr) else np.nan
            except Exception as e_mwr: metrics_entry['mwr'] = np.nan # Suppress detailed print
            # ... (Account total aggregation - unchanged) ...
            def safe_sum(df, col): return pd.to_numeric(df.get(col), errors='coerce').fillna(0.0).sum()
            metrics_entry['total_realized_gain_display'] = safe_sum(account_full_df, f'Realized Gain ({display_currency})'); metrics_entry['total_unrealized_gain_display'] = safe_sum(account_full_df, f'Unreal. Gain ({display_currency})'); metrics_entry['total_dividends_display'] = safe_sum(account_full_df, f'Dividends ({display_currency})'); metrics_entry['total_commissions_display'] = safe_sum(account_full_df, f'Commissions ({display_currency})'); metrics_entry['total_gain_display'] = safe_sum(account_full_df, f'Total Gain ({display_currency})'); metrics_entry['total_cash_display'] = safe_sum(account_full_df[account_full_df['Symbol'] == CASH_SYMBOL_CSV], f'Market Value ({display_currency})'); metrics_entry['total_cost_invested_display'] = safe_sum(account_full_df, f'Total Cost Invested ({display_currency})')
            acc_cumulative_investment_display = safe_sum(account_full_df, f'Cumulative Investment ({display_currency})')
            acc_total_gain = metrics_entry['total_gain_display']; acc_denominator = acc_cumulative_investment_display; acc_total_return_pct = np.nan
            if abs(acc_denominator) > 1e-9: acc_total_return_pct = (acc_total_gain / acc_denominator) * 100.0
            elif abs(acc_total_gain) <= 1e-9: acc_total_return_pct = 0.0
            metrics_entry['total_return_pct'] = acc_total_return_pct
            acc_total_day_change_display = safe_sum(account_full_df, f'Day Change ({display_currency})'); metrics_entry['total_day_change_display'] = acc_total_day_change_display
            acc_current_mv_display = metrics_entry['total_market_value_display']; acc_prev_close_mv_display = acc_current_mv_display - acc_total_day_change_display; metrics_entry['total_day_change_percent'] = np.nan
            if pd.notna(acc_total_day_change_display) and pd.notna(acc_prev_close_mv_display) and abs(acc_prev_close_mv_display) > 1e-9:
                 try: metrics_entry['total_day_change_percent'] = (acc_total_day_change_display / acc_prev_close_mv_display) * 100.0
                 except ZeroDivisionError: pass
            elif abs(acc_total_day_change_display) > 1e-9: metrics_entry['total_day_change_percent'] = np.inf if acc_total_day_change_display > 0 else -np.inf
            elif acc_total_day_change_display == 0 and acc_prev_close_mv_display == 0: metrics_entry['total_day_change_percent'] = 0.0

        # --- Overall metrics (unchanged logic, sums from filtered df) ---
        summary_df_for_return = full_summary_df
        if not show_closed_positions: original_count = len(full_summary_df); summary_df_for_return = full_summary_df[ full_summary_df['Quantity'].abs() > 1e-9 ].copy(); filtered_count = len(summary_df_for_return);
        overall_market_value_display = full_summary_df[f'Market Value ({display_currency})'].sum(); overall_cost_basis_display = full_summary_df.loc[ (full_summary_df['Quantity'].abs() > 1e-9) | (full_summary_df['Symbol'] == CASH_SYMBOL_CSV), f'Cost Basis ({display_currency})' ].sum(); overall_unrealized_gain_display = full_summary_df[f'Unreal. Gain ({display_currency})'].sum(); overall_realized_gain_display_agg = full_summary_df[f'Realized Gain ({display_currency})'].sum(); overall_dividends_display_agg = full_summary_df[f'Dividends ({display_currency})'].sum(); overall_commissions_display_agg = full_summary_df[f'Commissions ({display_currency})'].sum(); overall_total_gain_display = full_summary_df[f'Total Gain ({display_currency})'].sum(); overall_total_cost_invested_display = full_summary_df[f'Total Cost Invested ({display_currency})'].sum()
        overall_cumulative_investment_display = pd.to_numeric(full_summary_df[f'Cumulative Investment ({display_currency})'], errors='coerce').fillna(0.0).sum()
        try:
            # Pass the FILTERED transactions_df for overall MWR
            cf_dates_overall, cf_values_overall = get_cash_flows_for_mwr( transactions_df, overall_market_value_display, report_date, display_currency, current_fx_rates_vs_usd, display_currency )
            if cf_dates_overall and cf_values_overall: overall_portfolio_mwr = calculate_irr(cf_dates_overall, cf_values_overall)
        except Exception as e_mwr_overall: overall_portfolio_mwr = np.nan
        overall_day_change_display = full_summary_df[f'Day Change ({display_currency})'].sum(); overall_prev_close_mv_display = overall_market_value_display - overall_day_change_display; overall_day_change_percent = np.nan
        if pd.notna(overall_day_change_display) and pd.notna(overall_prev_close_mv_display) and abs(overall_prev_close_mv_display) > 1e-9:
             try: overall_day_change_percent = (overall_day_change_display / overall_prev_close_mv_display) * 100.0
             except ZeroDivisionError: pass
        elif abs(overall_day_change_display) > 1e-9: overall_day_change_percent = np.inf if overall_day_change_display > 0 else -np.inf
        elif abs(overall_day_change_display) < 1e-9 and abs(overall_prev_close_mv_display) < 1e-9: overall_day_change_percent = 0.0
        overall_total_return_pct = np.nan
        if abs(overall_cumulative_investment_display) > 1e-9: overall_total_return_pct = (overall_total_gain_display / overall_cumulative_investment_display) * 100.0
        elif abs(overall_total_gain_display) <= 1e-9: overall_total_return_pct = 0.0

        overall_summary_metrics = { "market_value": overall_market_value_display, "cost_basis_held": overall_cost_basis_display, "unrealized_gain": overall_unrealized_gain_display, "realized_gain": overall_realized_gain_display_agg, "dividends": overall_dividends_display_agg, "commissions": overall_commissions_display_agg, "total_gain": overall_total_gain_display, "total_cost_invested": overall_total_cost_invested_display, "portfolio_mwr": overall_portfolio_mwr * 100.0 if pd.notna(overall_portfolio_mwr) else np.nan, "day_change_display": overall_day_change_display if pd.notna(overall_day_change_display) else np.nan, "day_change_percent": overall_day_change_percent if pd.notna(overall_day_change_percent) else np.nan, "report_date": report_date.strftime('%Y-%m-%d'), "display_currency": display_currency, "cumulative_investment": overall_cumulative_investment_display, "total_return_pct": overall_total_return_pct }
        # Add available accounts list
        overall_summary_metrics['_available_accounts'] = all_available_accounts_list
        # Add exchange rate if needed
        if display_currency != default_currency:
            base_to_display_rate = get_conversion_rate(default_currency, display_currency, current_fx_rates_vs_usd)
            if base_to_display_rate != 1.0 and np.isfinite(base_to_display_rate):
                 overall_summary_metrics['exchange_rate_to_display'] = base_to_display_rate


    # --- 8. Prepare Ignored Transactions DataFrame (unchanged) ---
    # ... (Ignored DF preparation remains the same) ...
    ignored_df_final = pd.DataFrame()
    if ignored_indices:
        valid_indices = sorted([idx for idx in ignored_indices if idx in original_transactions_df.index])
        if valid_indices:
             ignored_df_final = original_transactions_df.loc[valid_indices].copy()
             try: ignored_df_final['Reason Ignored'] = ignored_df_final.index.map(ignored_reasons).fillna("Unknown Reason")
             except Exception as e_reason: print(f"Warning: Could not add 'Reason Ignored' column: {e_reason}")

    # --- 9. Determine Final Status (unchanged logic) ---
    # ... (Status determination remains the same) ...
    final_status = f"Success ({filter_desc})"
    price_source_warnings = False
    if not full_summary_df.empty and 'Price Source' in full_summary_df.columns:
         non_cash_holdings = full_summary_df[full_summary_df['Symbol'] != CASH_SYMBOL_CSV]
         price_source_warnings = non_cash_holdings['Price Source'].str.contains("Fallback|Excluded|Yahoo Invalid", na=False).any()
    warnings = any("WARN" in r.upper() for r in ignored_reasons.values()) or any("WARN" in s.upper() for s in status_messages) or any("MISSING" in r.upper() for r in ignored_reasons.values()) or price_source_warnings
    errors = any("ERROR" in r.upper() for r in ignored_reasons.values()) or any("FAIL" in r.upper() for r in ignored_reasons.values()) or any("ERROR" in s.upper() for s in status_messages) or (current_stock_data_internal is None or current_fx_rates_vs_usd is None)
    critical_warnings = any("CRITICAL" in r.upper() for r in ignored_reasons.values()) or any("CRITICAL" in s.upper() for s in status_messages)
    if critical_warnings: final_status = f"Success with Critical Warnings ({filter_desc})"
    elif errors: final_status = f"Success with Errors ({filter_desc})"
    elif warnings or ignored_indices: final_status = f"Success with Warnings ({filter_desc})"

    print(f"--- Portfolio Calculation Finished ({filter_desc}) ---")
    return overall_summary_metrics, summary_df_for_return, ignored_df_final, dict(account_level_metrics), final_status

# =======================================================================
# --- SECTION: HISTORICAL PERFORMANCE CALCULATION FUNCTIONS (REVISED + PARALLEL + CACHE) ---
# =======================================================================

# --- Function to map internal symbols to Yahoo Finance format ---
def map_to_yf_symbol(internal_symbol: str) -> Optional[str]:
    """Maps an internal symbol to a Yahoo Finance compatible ticker."""
    if internal_symbol == CASH_SYMBOL_CSV or internal_symbol in YFINANCE_EXCLUDED_SYMBOLS: return None
    if internal_symbol in SYMBOL_MAP_TO_YFINANCE: return SYMBOL_MAP_TO_YFINANCE[internal_symbol]
    if internal_symbol.endswith(':BKK'):
        base_symbol = internal_symbol[:-4];
        if base_symbol in SYMBOL_MAP_TO_YFINANCE: base_symbol = SYMBOL_MAP_TO_YFINANCE[base_symbol]
        if '.' in base_symbol or len(base_symbol) == 0: print(f"Hist WARN: Skipping potentially invalid BKK conversion: {internal_symbol}"); return None
        return f"{base_symbol.upper()}.BK"
    if ' ' in internal_symbol or any(c in internal_symbol for c in [':', ',']): print(f"Hist WARN: Skipping potentially invalid symbol format for YF: {internal_symbol}"); return None
    return internal_symbol.upper()

# --- Yahoo Finance Historical Data Fetching ---
def fetch_yf_historical(symbols_yf: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
    """ Fetches historical 'Close' data (automatically adjusted for splits AND dividends) for multiple symbols from Yahoo Finance. """
    if not YFINANCE_AVAILABLE: print("Error: yfinance not available for historical fetch."); return {}
    historical_data: Dict[str, pd.DataFrame] = {}
    if not symbols_yf: print("Hist Fetch: No symbols provided."); return historical_data
    print(f"Hist Fetch: Fetching historical data (auto-adjusted) for {len(symbols_yf)} symbols from Yahoo Finance ({start_date} to {end_date})...")
    yf_end_date = max(start_date, end_date) + timedelta(days=1); yf_start_date = min(start_date, end_date)
    fetch_batch_size = 50; symbols_processed = 0
    for i in range(0, len(symbols_yf), fetch_batch_size):
        batch_symbols = symbols_yf[i:i + fetch_batch_size]
        try:
            data = yf.download( tickers=batch_symbols, start=yf_start_date, end=yf_end_date, progress=False, group_by='ticker', auto_adjust=True, actions=False )
            if data.empty: # print(f"  Hist WARN: No data returned for batch: {', '.join(batch_symbols)}"); # Reduced verbosity
                continue
            for symbol in batch_symbols:
                df_symbol = None
                try:
                    if len(batch_symbols) == 1 and not data.columns.nlevels > 1: df_symbol = data if not data.empty else None
                    elif symbol in data.columns.levels[0]: df_symbol = data[symbol]
                    elif len(batch_symbols) > 1 and not data.columns.nlevels > 1: # print(f"  Hist WARN: Unexpected flat DataFrame structure for multi-ticker batch with auto_adjust=True. Symbol {symbol} might be missing."); # Reduced verbosity
                         continue
                    else:
                         if isinstance(data, pd.Series) and data.name == symbol: df_symbol = pd.DataFrame(data)
                         elif isinstance(data, pd.DataFrame) and symbol in data.columns: df_symbol = data[[symbol]].rename(columns={symbol:'Close'})
                         else: # print(f"  Hist WARN: Symbol {symbol} not found in yfinance download results for this batch (Structure: {data.columns})."); # Reduced verbosity
                              continue
                    if df_symbol is None or df_symbol.empty: continue
                    price_col = 'Close'
                    if price_col not in df_symbol.columns: # print(f"  Hist WARN: Expected 'Close' column not found for {symbol}. Columns: {df_symbol.columns}"); # Reduced verbosity
                         continue
                    df_filtered = df_symbol[[price_col]].copy(); df_filtered.rename(columns={price_col: 'price'}, inplace=True)
                    df_filtered.index = pd.to_datetime(df_filtered.index).date
                    df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce'); df_filtered = df_filtered.dropna(subset=['price']); df_filtered = df_filtered[df_filtered['price'] > 1e-6]
                    if not df_filtered.empty: historical_data[symbol] = df_filtered.sort_index()
                except Exception as e_sym: print(f"  Hist ERROR processing symbol {symbol} within batch: {e_sym}")
        except Exception as e_batch: print(f"  Hist ERROR during yf.download for batch starting with {batch_symbols[0]}: {e_batch}")
        symbols_processed += len(batch_symbols); time.sleep(0.2)
    print(f"Hist Fetch: Finished fetching ({len(historical_data)} symbols successful).")
    return historical_data

# --- Function to Unadjust Prices based on Splits ---
def _unadjust_prices(
    adjusted_prices_yf: Dict[str, pd.DataFrame], # Key: YF symbol, Val: DF with 'price' (ADJUSTED)
    yf_to_internal_map: Dict[str, str],          # Map YF symbol back to internal
    splits_by_internal_symbol: Dict[str, List[Dict]], # Internal symbol -> List of {'Date': date, 'Split Ratio': float}
    processed_warnings: set # To avoid spamming warnings
) -> Dict[str, pd.DataFrame]:
    """
    Derives unadjusted prices from adjusted prices using split history.
    Uses the formula: Unadjusted = Adjusted * Cumulative_Forward_Split_Factor.
    """
    unadjusted_prices_yf = {}
    unadjusted_count = 0

    for yf_symbol, adj_price_df in adjusted_prices_yf.items():
        if adj_price_df.empty or 'price' not in adj_price_df.columns:
            unadjusted_prices_yf[yf_symbol] = adj_price_df.copy()
            continue

        internal_symbol = yf_to_internal_map.get(yf_symbol)
        symbol_splits = None
        if internal_symbol:
            symbol_splits = splits_by_internal_symbol.get(internal_symbol)

        if not symbol_splits:
            unadjusted_prices_yf[yf_symbol] = adj_price_df.copy()
            continue

        unadj_df = adj_price_df.copy()
        if not isinstance(unadj_df.index, pd.DatetimeIndex):
             try:
                 # Ensure index is consistently date objects
                 unadj_df.index = pd.to_datetime(unadj_df.index, errors='coerce').date
                 unadj_df = unadj_df[pd.notnull(unadj_df.index)] # Drop NaT indices
             except Exception:
                 warn_key = f"unadjust_index_err_{yf_symbol}";
                 if warn_key not in processed_warnings: print(f"  Hist WARN (Unadjust): Failed to convert index to date for {yf_symbol}. Skipping unadjustment."); processed_warnings.add(warn_key)
                 unadjusted_prices_yf[yf_symbol] = adj_price_df.copy(); continue

        if unadj_df.empty: unadjusted_prices_yf[yf_symbol] = unadj_df; continue
        unadj_df.sort_index(inplace=True)

        forward_split_factor = pd.Series(1.0, index=unadj_df.index, dtype=float)
        # Ensure splits are sorted correctly (most recent first)
        sorted_splits_desc = sorted(symbol_splits, key=lambda x: x.get('Date', date.min), reverse=True) # Added .get with default

        for split_info in sorted_splits_desc:
            try:
                split_date_raw = split_info.get('Date')
                if split_date_raw is None: raise ValueError("Split info missing 'Date'")
                # Ensure split_date is a date object
                if isinstance(split_date_raw, datetime): split_date = split_date_raw.date()
                elif isinstance(split_date_raw, date): split_date = split_date_raw
                else: raise TypeError(f"Split date is not a valid date or datetime object: {type(split_date_raw)}")

                split_ratio = float(split_info['Split Ratio'])

                if split_ratio <= 0:
                    warn_key = f"invalid_split_ratio_{yf_symbol}_{split_date}";
                    if warn_key not in processed_warnings: print(f"  Hist WARN (Unadjust): Invalid split ratio ({split_ratio}) for {yf_symbol} on {split_date}. Skipping this split."); processed_warnings.add(warn_key)
                    continue
                # Apply split factor to dates *before* the split date
                mask = forward_split_factor.index < split_date
                forward_split_factor.loc[mask] *= split_ratio
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                 warn_key = f"split_error_{yf_symbol}_{split_info.get('Date', 'UnknownDate')}"
                 error_detail = f"{type(e).__name__}: {e}"
                 if warn_key not in processed_warnings: print(f"  Hist WARN (Unadjust): Error processing split for {yf_symbol} around {split_info.get('Date', 'UnknownDate')}: {error_detail}. Skipping."); processed_warnings.add(warn_key)
                 continue

        original_prices = unadj_df['price'].copy()
        # Align series before multiplication, handle missing dates
        aligned_factor, aligned_prices = forward_split_factor.align(unadj_df['price'], join='right', fill_value=1.0)
        unadj_df['price'] = aligned_prices * aligned_factor

        if not unadj_df['price'].equals(original_prices.reindex_like(unadj_df['price'])): unadjusted_count += 1
        unadjusted_prices_yf[yf_symbol] = unadj_df

    return unadjusted_prices_yf

# --- Helper Functions for Point-in-Time Historical Calculation ---
def get_historical_price(symbol_key: str, target_date: date, prices_dict: Dict[str, pd.DataFrame]) -> Optional[float]:
    """Gets the price for a symbol (YF ticker/pair) on a date, using forward fill."""
    if symbol_key not in prices_dict or prices_dict[symbol_key].empty: return None
    df = prices_dict[symbol_key]
    try:
        # Ensure df index is date objects
        if not all(isinstance(idx, date) for idx in df.index):
             # Attempt conversion if needed
             df.index = pd.to_datetime(df.index).date

        # Ensure target_date is a date object
        if not isinstance(target_date, date): return None

        # Efficient lookup using reindex with forward fill
        # Make sure index is sorted before reindexing
        if not df.index.is_monotonic_increasing: df.sort_index(inplace=True)
        combined_index = df.index.union([target_date])
        df_reindexed = df.reindex(combined_index, method='ffill')
        price = df_reindexed.loc[target_date]['price']

        # If target_date is before the first price, ffill gives NaN, which is correct
        if pd.isna(price) and target_date < df.index.min(): return None
        return float(price) if pd.notna(price) else None
    except KeyError: return None
    except Exception as e:
        # print(f"ERROR getting historical price for {symbol_key} on {target_date}: {e}") # Reduced verbosity
        return None


# --- Revised: Calculates historical rate TO/FROM via USD bridge ---
# --- FINAL Version: Calculates historical rate TO/FROM via USD bridge ---
def get_historical_fx_rate(from_curr: str, to_curr: str, target_date: date, fx_dict: Dict[str, pd.DataFrame]) -> float:
    """
    Gets the historical FX rate (units of to_curr per 1 unit of from_curr)
    using Yahoo Finance pairs (e.g., EURUSD=X means USD per 1 EUR).
    Calculates cross rates via USD. Returns np.nan if rate cannot be found/derived.
    """
    if from_curr == to_curr:
        return 1.0 # Rate is 1 if currencies are the same
    if not from_curr or not to_curr or from_curr == 'N/A' or to_curr == 'N/A' or not isinstance(target_date, date):
        return np.nan # Invalid input

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()
    rate_found = np.nan

    # --- Calculate via USD: Target Rate = TO / FROM ---
    # Formula derived: TO / FROM = value(USD per FROM) / value(USD per TO)

    # Get value for USD per 1 FROM currency (e.g., USD/THB)
    # Yahoo pair needed: FROMUSD=X (e.g., THBUSD=X -> gives USD per 1 THB)
    val_usd_per_from = np.nan
    if from_curr_upper == 'USD':
        val_usd_per_from = 1.0
    else:
        pair_from_usd_yahoo = f"{from_curr_upper}USD=X" # e.g., THBUSD=X
        price = get_historical_price(pair_from_usd_yahoo, target_date, fx_dict)
        if price is not None and pd.notna(price):
            # Check for potentially zero rate from source before assigning
            if abs(price) > 1e-9:
                 val_usd_per_from = price # This directly gives USD / FROM
            # else: Treat zero rate as invalid -> Keep NaN

    # Get value for USD per 1 TO currency (e.g., USD/JPY)
    # Yahoo pair needed: TOUSD=X (e.g., JPYUSD=X -> gives USD per 1 JPY)
    val_usd_per_to = np.nan
    if to_curr_upper == 'USD':
        val_usd_per_to = 1.0
    else:
        pair_to_usd_yahoo = f"{to_curr_upper}USD=X" # e.g., JPYUSD=X
        price = get_historical_price(pair_to_usd_yahoo, target_date, fx_dict)
        if price is not None and pd.notna(price):
             # Check for potentially zero rate from source before assigning
             if abs(price) > 1e-9:
                  val_usd_per_to = price # This directly gives USD / TO
             # else: Treat zero rate as invalid -> Keep NaN

    # Calculate final rate TO / FROM = val_usd_per_from / val_usd_per_to
    # Calculation: (USD / FROM) / (USD / TO) = (USD/FROM) * (TO/USD) = TO / FROM
    if pd.notna(val_usd_per_from) and pd.notna(val_usd_per_to):
        if abs(val_usd_per_to) > 1e-9: # Check denominator is not effectively zero
            try:
                rate_found = val_usd_per_from / val_usd_per_to # Correctly yields TO / FROM
            except ZeroDivisionError:
                rate_found = np.nan
        else:
            rate_found = np.nan # Denominator is zero/invalid
    else:
        rate_found = np.nan # Failed to get intermediate rates

    # Return NaN on failure, allows calling function to know lookup failed
    return float(rate_found) if pd.notna(rate_found) else np.nan

# --- Function to Calculate Daily Net External Cash Flow ---
def get_historical_fx_rate(from_curr: str, to_curr: str, target_date: date, fx_dict: Dict[str, pd.DataFrame]) -> float:
    """
    Gets the historical FX rate using Yahoo Finance pairs (e.g., EURUSD=X).
    Prioritizes standard pairs (relative to USD) and calculates inverses correctly.
    Returns np.nan if the rate cannot be found or derived.
    """
    if from_curr == to_curr: return 1.0
    if not from_curr or not to_curr or from_curr == 'N/A' or to_curr == 'N/A': return np.nan

    from_curr_upper = from_curr.upper(); to_curr_upper = to_curr.upper()
    rate_found = np.nan # Default to NaN

    # Strategy:
    # 1. Try direct pair (e.g., THBUSD=X)
    # 2. Try inverse pair (e.g., USDTHB=X) and calculate 1/rate
    # 3. Try via USD (e.g., for THB/EUR, get THBUSD=X and EURUSD=X, calculate THBUSD / EURUSD) - **More robust**

    # --- Revised Strategy (via USD) ---
    rate_usd_per_from = 1.0 if from_curr_upper == 'USD' else np.nan
    rate_usd_per_to = 1.0 if to_curr_upper == 'USD' else np.nan

    # Get From -> USD rate
    if from_curr_upper != 'USD':
        pair_from_usd = f"{from_curr_upper}USD=X"
        pair_usd_from = f"USD{from_curr_upper}=X"
        # Prefer FromUSD=X
        val = get_historical_price(pair_from_usd, target_date, fx_dict)
        if val is not None and pd.notna(val) and val > 1e-9:
            rate_usd_per_from = 1.0 / val # We want USD per From
        else: # Try USDFrom=X
            val_inv = get_historical_price(pair_usd_from, target_date, fx_dict)
            if val_inv is not None and pd.notna(val_inv) and val_inv > 1e-9:
                rate_usd_per_from = val_inv

    # Get To -> USD rate
    if to_curr_upper != 'USD':
        pair_to_usd = f"{to_curr_upper}USD=X"
        pair_usd_to = f"USD{to_curr_upper}=X"
        # Prefer ToUSD=X
        val = get_historical_price(pair_to_usd, target_date, fx_dict)
        if val is not None and pd.notna(val) and val > 1e-9:
            rate_usd_per_to = 1.0 / val # We want USD per To
        else: # Try USDTo=X
             val_inv = get_historical_price(pair_usd_to, target_date, fx_dict)
             if val_inv is not None and pd.notna(val_inv) and val_inv > 1e-9:
                 rate_usd_per_to = val_inv

    # Calculate cross rate (To / From, relative to USD)
    if pd.notna(rate_usd_per_from) and pd.notna(rate_usd_per_to):
        if abs(rate_usd_per_from) > 1e-9:
            try:
                rate_found = rate_usd_per_to / rate_usd_per_from
            except ZeroDivisionError:
                rate_found = np.nan
        else: rate_found = np.nan
    else:
        rate_found = np.nan # Failed to get one of the USD rates

    return float(rate_found) if pd.notna(rate_found) else np.nan

def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame, # Assumes 'Local Currency' column exists
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str], # <-- Added
    default_currency: str,              # <-- Added
    processed_warnings: set
) -> Tuple[float, bool]: # Returns (net_flow, fx_lookup_failed)
    """
    Calculates the net *EXTERNAL* cash flow for a specific day in the target currency.
    External flows are ONLY transactions explicitly marked as 'deposit' or 'withdrawal'.
    """
    fx_lookup_failed = False
    net_flow_target_curr = 0.0
    daily_tx = transactions_df[transactions_df['Date'].dt.date == target_date].copy()
    if daily_tx.empty: return 0.0, False

    cash_flow_tx = daily_tx[ (daily_tx['Symbol'] == CASH_SYMBOL_CSV) & (daily_tx['Type'].isin(['deposit', 'withdrawal'])) ].copy()
    if cash_flow_tx.empty: return 0.0, False

    for _, row in cash_flow_tx.iterrows():
        tx_type = row['Type']
        qty = pd.to_numeric(row['Quantity'], errors='coerce')
        commission_local_raw = pd.to_numeric(row.get('Commission'), errors='coerce')
        commission_local = 0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        # --- Use Local Currency from DataFrame ---
        local_currency = row['Local Currency']
        # ---------------------------------------
        flow_local = 0.0

        if pd.isna(qty): # Skip if qty missing
            warn_key = f"missing_cash_qty_{target_date}"; # Suppressed warning
            continue

        if tx_type == 'deposit': flow_local = abs(qty) - commission_local
        elif tx_type == 'withdrawal': flow_local = -abs(qty) - commission_local

        flow_target = flow_local
        if local_currency != target_currency:
            fx_rate = get_historical_fx_rate(local_currency, target_currency, target_date, historical_fx_yf)
            if pd.isna(fx_rate):
                 warn_key = f"missing_fx_cashflow_{target_date}_{local_currency}_{target_currency}"; # Suppressed warning
                 fx_lookup_failed = True; flow_target = np.nan
            else: flow_target = flow_local * fx_rate

        if pd.notna(flow_target): net_flow_target_curr += flow_target
        else: return np.nan, True

    return net_flow_target_curr, fx_lookup_failed

# --- Point-in-Time Portfolio Value Calculation (Uses UNADJUSTED for portfolio, needed by worker) ---
def _calculate_portfolio_value_at_date_unadjusted(
    target_date: date,
    transactions_df: pd.DataFrame, # Assumes 'Local Currency' column exists
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str], # <-- Added
    default_currency: str,              # <-- Added
    processed_warnings: set
) -> Tuple[float, bool]: # Returns (calculated_value, lookup_failed)
    """
    Calculates the total portfolio market value for a specific date using
    UNADJUSTED historical prices for stocks/ETFs and handling cash balances.
    Uses the 'Local Currency' column from transactions_df.
    Includes corrected price lookup logic.
    """
    transactions_til_date = transactions_df[transactions_df['Date'].dt.date <= target_date].copy()
    if transactions_til_date.empty: return 0.0, False

    holdings: Dict[Tuple[str, str], Dict] = {}
    # --- Holdings quantity calculation loop (uses Local Currency from df) ---
    for index, row in transactions_til_date.iterrows():
        symbol = str(row.get('Symbol', 'UNKNOWN')).strip(); account = str(row.get('Account', 'Unknown'))
        local_currency = str(row.get('Local Currency', default_currency))
        holding_key = (symbol, account); tx_type = str(row.get('Type', 'UNKNOWN_TYPE')).lower().strip()
        if symbol == CASH_SYMBOL_CSV: continue
        if holding_key not in holdings: holdings[holding_key] = { 'qty': 0.0, 'local_currency': local_currency, 'is_stock': True }
        elif holdings[holding_key]['local_currency'] != local_currency: holdings[holding_key]['local_currency'] = local_currency
        try:
            holding = holdings[holding_key]; qty = pd.to_numeric(row.get('Quantity'), errors='coerce'); split_ratio = pd.to_numeric(row.get('Split Ratio'), errors='coerce')
            if symbol in SHORTABLE_SYMBOLS and tx_type in ['short sell', 'buy to cover']:
                 if pd.isna(qty): continue
                 qty_abs = abs(qty)
                 if tx_type == 'short sell': holding['qty'] -= qty_abs
                 elif tx_type == 'buy to cover': current_short_qty_abs = abs(holding['qty']) if holding['qty'] < -1e-9 else 0.0; qty_being_covered = min(qty_abs, current_short_qty_abs); holding['qty'] += qty_being_covered
            elif tx_type == 'buy' or tx_type == 'deposit':
                if pd.notna(qty) and qty > 0: holding['qty'] += qty
            elif tx_type == 'sell' or tx_type == 'withdrawal':
                if pd.notna(qty) and qty > 0: sell_qty = qty; held_qty = holding['qty']; qty_sold = min(sell_qty, held_qty) if held_qty > 1e-9 else 0; holding['qty'] -= qty_sold
            elif tx_type in ['split', 'stock split']:
                if pd.notna(split_ratio) and split_ratio > 0 and abs(holding['qty']) > 1e-9: holding['qty'] *= split_ratio;
                if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0
        except Exception as e_h: pass

    # --- Cash balance calculation (uses Local Currency from df) ---
    cash_summary: Dict[str, Dict] = {}
    cash_transactions = transactions_til_date[transactions_til_date['Symbol'] == CASH_SYMBOL_CSV].copy()
    if not cash_transactions.empty:
        def get_signed_quantity_cash(row): type_lower = str(row.get('Type', '')).lower(); qty = pd.to_numeric(row.get('Quantity'), errors='coerce'); return 0.0 if pd.isna(qty) else (abs(qty) if type_lower in ['buy', 'deposit'] else (-abs(qty) if type_lower in ['sell', 'withdrawal'] else 0.0))
        cash_transactions['SignedQuantity'] = cash_transactions.apply(get_signed_quantity_cash, axis=1)
        cash_qty_agg = cash_transactions.groupby('Account')['SignedQuantity'].sum()
        cash_currency_map = cash_transactions.groupby('Account')['Local Currency'].first()
        all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index)
        for acc in all_cash_accounts: cash_summary[acc] = { 'qty': cash_qty_agg.get(acc, 0.0), 'local_currency': cash_currency_map.get(acc, default_currency), 'is_stock': False }

    # --- Combine stock and cash positions (unchanged) ---
    all_positions: Dict[Tuple[str, str], Dict] = {**holdings, **{(CASH_SYMBOL_CSV, acc): data for acc, data in cash_summary.items()}}

    # --- Calculate Total Market Value (uses local_currency from combined positions) ---
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False

    for (internal_symbol, account), data in all_positions.items():
        current_qty = data.get('qty', 0.0)
        local_currency = data.get('local_currency', default_currency)
        is_stock = data.get('is_stock', internal_symbol != CASH_SYMBOL_CSV)

        if abs(current_qty) < 1e-9: continue # Skip zero quantity holdings

        # --- Step 1: Get FX Rate ---
        fx_rate = get_historical_fx_rate(local_currency, target_currency, target_date, historical_fx_yf)
        if pd.isna(fx_rate):
            warn_key = f"missing_fx_value_{target_date}_{local_currency}_{target_currency}_{internal_symbol}" # Suppressed warning
            any_lookup_nan_on_date = True; total_market_value_display_curr_agg = np.nan; break # Critical failure for the date

        # --- Step 2: Determine Local Price (with fallback) ---
        current_price_local = np.nan # Initialize
        force_fallback = (internal_symbol in YFINANCE_EXCLUDED_SYMBOLS)
        price_val = None # <<< FIX: Initialize price_val here

        if not is_stock:
            current_price_local = 1.0 # Cash
        elif not force_fallback: # Stock/ETF not excluded - Attempt primary lookup
            yf_symbol_for_lookup = internal_to_yf_map.get(internal_symbol)
            if yf_symbol_for_lookup:
                price_val = get_historical_price(yf_symbol_for_lookup, target_date, historical_prices_yf_unadjusted)
                # <<< FIX: Nest the check for price_val inside the 'if yf_symbol_for_lookup' block >>>
                if price_val is not None and pd.notna(price_val) and price_val > 1e-9:
                    current_price_local = float(price_val)
                # else: current_price_local remains NaN, fallback will be attempted below

        # --- Fallback Logic ---
        # Trigger fallback if primary lookup failed (current_price_local is NaN) OR if fallback was forced
        if pd.isna(current_price_local) or force_fallback:
             try:
                 # Find last transaction price ON OR BEFORE target_date
                 fallback_tx = transactions_df[
                     (transactions_df['Symbol'] == internal_symbol) &
                     (transactions_df['Account'] == account) &
                     (transactions_df['Price/Share'].notna()) &
                     (transactions_df['Price/Share'] > 1e-9) &
                     (transactions_df['Date'].dt.date <= target_date)
                 ].copy()

                 if not fallback_tx.empty:
                     fallback_tx.sort_values(by=['Date', 'original_index'], inplace=True, ascending=True)
                     last_tx_row = fallback_tx.iloc[-1]
                     last_tx_price = pd.to_numeric(last_tx_row['Price/Share'], errors='coerce')

                     # Use fallback only if primary lookup failed or was forced
                     # Check pd.isna(current_price_local) again to ensure we only overwrite if needed
                     if pd.notna(last_tx_price) and last_tx_price > 1e-9 and (pd.isna(current_price_local) or force_fallback):
                         current_price_local = float(last_tx_price)
                     # else: price remains NaN if last tx price was invalid/non-positive

             except Exception as e_fallback:
                 # Suppress warning/error print in worker for performance
                 pass # Keep current_price_local as NaN

        # --- Step 3: Check if Price Determination Failed ---
        if pd.isna(current_price_local):
             warn_key = f"final_price_nan_{target_date}_{internal_symbol}" # Suppressed warning
             any_lookup_nan_on_date = True; total_market_value_display_curr_agg = np.nan; break # Critical failure for the date

        # --- Step 4: Calculate Market Value ---
        # Ensure current_price_local is float before multiplication
        market_value_local = current_qty * float(current_price_local)
        market_value_display = market_value_local * fx_rate

        # --- Step 5: Aggregate ---
        if pd.isna(market_value_display):
             any_lookup_nan_on_date = True; total_market_value_display_curr_agg = np.nan; break
        else:
             total_market_value_display_curr_agg += market_value_display
    # --- End Loop over positions ---

    return total_market_value_display_curr_agg, any_lookup_nan_on_date

# --- Worker function for multiprocessing ---
def _calculate_daily_metrics_worker(
    eval_date: date,
    # --- Passed via functools.partial ---
    transactions_df: pd.DataFrame, # Assumes 'Local Currency' column exists
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str], # <-- Added
    default_currency: str,              # <-- Added
    benchmark_symbols_yf: List[str]
) -> Optional[Dict]:
    """ Worker function to calculate portfolio value, net flow, and benchmark prices for a single date. """
    try:
        dummy_warnings_set = set()
        # 1. Calculate Portfolio Value (Pass map/default)
        portfolio_value, val_lookup_failed = _calculate_portfolio_value_at_date_unadjusted(
            eval_date, transactions_df, historical_prices_yf_unadjusted,
            historical_fx_yf, target_currency, internal_to_yf_map,
            account_currency_map, default_currency, dummy_warnings_set # Pass map/default
        )
        # 2. Calculate Net External Cash Flow (Pass map/default)
        net_cash_flow, flow_lookup_failed = _calculate_daily_net_cash_flow(
            eval_date, transactions_df, target_currency, historical_fx_yf,
            account_currency_map, default_currency, dummy_warnings_set # Pass map/default
        )
        if pd.isna(portfolio_value): net_cash_flow = np.nan; val_lookup_failed = True
        # 3. Get benchmark prices (unchanged)
        benchmark_prices = {}; bench_lookup_failed = False
        for bm_symbol in benchmark_symbols_yf:
            price = get_historical_price(bm_symbol, eval_date, historical_prices_yf_adjusted)
            bench_price = float(price) if pd.notna(price) else np.nan; benchmark_prices[f"{bm_symbol} Price"] = bench_price
            if pd.isna(bench_price): bench_lookup_failed = True
        # 4. Assemble result (unchanged)
        result_row = { 'Date': eval_date, 'value': portfolio_value, 'net_flow': net_cash_flow, 'value_lookup_failed': val_lookup_failed, 'flow_lookup_failed': flow_lookup_failed, 'bench_lookup_failed': bench_lookup_failed }
        result_row.update(benchmark_prices)
        return result_row
    except Exception as e: # Error handling unchanged
        # ... (Error logging and return structure) ...
        print(f"ERROR in worker process for date {eval_date}: {e}"); traceback.print_exc()
        failed_row = { 'Date': eval_date, 'value': np.nan, 'net_flow': np.nan };
        for bm_symbol in benchmark_symbols_yf: failed_row[f"{bm_symbol} Price"] = np.nan
        failed_row['value_lookup_failed'] = True; failed_row['flow_lookup_failed'] = True; failed_row['bench_lookup_failed'] = True; failed_row['worker_error'] = True
        return failed_row

# --- Historical Performance Calculation Wrapper Function (Refactored for Parallelism + Results Cache) ---
def calculate_historical_performance(
    transactions_csv_file: str,
    start_date: date,
    end_date: date,
    interval: str, # 'D', 'W', 'M'
    benchmark_symbols_yf: List[str],
    display_currency: str,
    account_currency_map: Dict,
    default_currency: str,
    use_raw_data_cache: bool = True,
    use_daily_results_cache: bool = True,
    num_processes: Optional[int] = None,
    include_accounts: Optional[List[str]] = None, # <-- NEW Parameter, replaces account_filter
    exclude_accounts: Optional[List[str]] = None # <-- ADDED exclude_accounts parameter
) -> Tuple[pd.DataFrame, str]:
    """
    Calculates historical portfolio performance (TWR-like accumulated gain)
    and benchmark performance over a date range using parallel daily calculations.
    Filters calculations based on the `include_accounts` and `exclude_accounts` lists.
    Caches results based on the effective set of accounts used.

    Args:
       transactions_csv_file (str): Path to the transactions CSV file.
       start_date (date): Start date for the analysis.
       end_date (date): End date for the analysis.
       interval (str): Resampling interval ('D', 'W', 'M').
       benchmark_symbols_yf (List[str]): List of Yahoo Finance benchmark tickers.
       display_currency (str): Currency for portfolio value calculation.
       account_currency_map (Dict): Map of account names to their local currency.
       default_currency (str): Default currency if account not in map.
       use_raw_data_cache (bool): Whether to use/save cache for raw historical prices/FX.
       use_daily_results_cache (bool): Whether to use/save cache for calculated daily results.
       num_processes (Optional[int]): Number of parallel processes. Defaults to CPU count.
       include_accounts (Optional[List[str]]): List of account names to include. If None or empty, all accounts are included.
       exclude_accounts (Optional[List[str]]): List of account names to exclude from the included set.

    Returns:
        Tuple[pd.DataFrame, str]:
            - DataFrame index 'Date', columns including 'Portfolio Value', 'Portfolio Daily Gain', 'daily_return',
              'Portfolio Accumulated Gain', '{Bench} Price' (Adjusted), '{Bench} Accumulated Gain'.
              The portfolio columns represent the effective accounts scope (included - excluded).
              Returns an empty DataFrame on critical failure.
            - Status message string indicating success, warnings, or errors, potentially including final TWR factor.
    """
    # --- Versioning and Cache Path (v10) ---
    CURRENT_HIST_VERSION = "v10"
    DAILY_RESULTS_CACHE_PATH_PREFIX = f'yf_portfolio_daily_results_{CURRENT_HIST_VERSION}'

    # --- Determine effective filter description ---
    # --- MODIFICATION: Update filter description based on include/exclude ---
    filter_desc = "All Accounts"
    if include_accounts:
        filter_desc = f"Included: {', '.join(sorted(include_accounts))}"
    if exclude_accounts:
        if include_accounts: # If both are specified
             filter_desc += f" (Excluding: {', '.join(sorted(exclude_accounts))})"
        else: # Only exclusion specified (means exclude from all)
             filter_desc = f"All Accounts (Excluding: {', '.join(sorted(exclude_accounts))})"
    # --- END MODIFICATION ---

    print(f"\n--- Starting Historical Performance Calculation ({CURRENT_HIST_VERSION} - Scope: {filter_desc}) ---")
    print(f"Params: CSV='{os.path.basename(transactions_csv_file)}', Start={start_date}, End={end_date}, Interval={interval}")
    print(f"        Benchmarks={benchmark_symbols_yf}, Currency='{display_currency}', RawCache={use_raw_data_cache}, DailyResultsCache={use_daily_results_cache}")

    start_time_hist = time.time()
    status_msg = f"Historical ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Initializing..."
    processed_warnings = set()
    final_twr_factor = np.nan

    # --- 0. Initial Checks (unchanged) ---
    if not YFINANCE_AVAILABLE: return pd.DataFrame(), "Error: yfinance library not installed."
    if start_date >= end_date: return pd.DataFrame(), "Error: Start date must be before end date."
    if interval not in ['D', 'W', 'M']: return pd.DataFrame(), f"Error: Invalid interval '{interval}'."
    clean_benchmark_symbols_yf = []
    if benchmark_symbols_yf and isinstance(benchmark_symbols_yf, list):
        clean_benchmark_symbols_yf = [b.upper().strip() for b in benchmark_symbols_yf if isinstance(b, str) and b.strip()]
    elif not benchmark_symbols_yf:
        print(f"Hist INFO ({CURRENT_HIST_VERSION}): No benchmark symbols provided.")
    else:
        return pd.DataFrame(), "Error: Invalid list of benchmark symbols provided."

    # --- 1. Load & Clean ALL Transactions (unchanged) ---
    all_transactions_df, original_transactions_df, ignored_indices, ignored_reasons = _load_and_clean_transactions(
        transactions_csv_file, account_currency_map, default_currency
    )
    if all_transactions_df is None: return pd.DataFrame(), f"Error: Failed to load/clean transactions from '{transactions_csv_file}'."
    if ignored_reasons: status_msg += f" Tx loaded ({len(ignored_reasons)} ignored)."
    else: status_msg += " Tx loaded."

    # --- 1b. Filter Transactions based on include_accounts AND exclude_accounts ---
    transactions_df_effective = pd.DataFrame() # This will hold the final set of transactions
    available_accounts = set(all_transactions_df['Account'].unique()) if 'Account' in all_transactions_df.columns else set()
    included_accounts_list_sorted = [] # For cache key
    excluded_accounts_list_sorted = [] # For cache key

    # Step 1: Apply inclusion filter
    if not include_accounts: # If None or empty list passed, start with all
        print(f"Hist ({CURRENT_HIST_VERSION}): No specific accounts provided for inclusion, starting with all available accounts.")
        transactions_df_included = all_transactions_df.copy()
        included_accounts_list_sorted = sorted(list(available_accounts))
    else:
        valid_include_accounts = [acc for acc in include_accounts if acc in available_accounts]
        if not valid_include_accounts:
            msg = f"Hist WARN ({CURRENT_HIST_VERSION}): None of the specified accounts to include were found in transactions. No data to process."
            print(msg)
            return pd.DataFrame(), f"Historical ({CURRENT_HIST_VERSION}): Success (No data for specified accounts)."
        print(f"Hist ({CURRENT_HIST_VERSION}): Filtering transactions FOR accounts: {', '.join(sorted(valid_include_accounts))}")
        transactions_df_included = all_transactions_df[all_transactions_df['Account'].isin(valid_include_accounts)].copy()
        included_accounts_list_sorted = sorted(valid_include_accounts)

    # Step 2: Apply exclusion filter to the included set
    if not exclude_accounts or not isinstance(exclude_accounts, list):
        # No exclusion needed, the included set is the effective set
        transactions_df_effective = transactions_df_included.copy()
        excluded_accounts_list_sorted = [] # Empty list for cache key
    else:
        # Remove accounts specified in exclude_accounts
        valid_exclude_accounts = [acc for acc in exclude_accounts if acc in available_accounts] # Use available_accounts derived from *all* transactions
        if not valid_exclude_accounts:
            # No valid accounts to exclude, effective set is the included set
            transactions_df_effective = transactions_df_included.copy()
            excluded_accounts_list_sorted = []
        else:
            # --- START MODIFICATION ---
            print(f"Hist ({CURRENT_HIST_VERSION}): Further filtering transactions, EXCLUDING accounts: {', '.join(sorted(valid_exclude_accounts))}")
            # Filter the already included dataframe
            transactions_df_effective = transactions_df_included[~transactions_df_included['Account'].isin(valid_exclude_accounts)].copy()
            excluded_accounts_list_sorted = sorted(valid_exclude_accounts)
            # --- END MODIFICATION ---
            
    # Check if any transactions remain after all filtering
    if transactions_df_effective.empty:
        msg = f"Hist WARN ({CURRENT_HIST_VERSION}): No transactions remain after applying inclusion/exclusion filters."
        print(msg)
        return pd.DataFrame(), f"Historical ({CURRENT_HIST_VERSION}): Success (No data after filtering)."
    # --- END MODIFICATION for include/exclude filtering ---

    # --- Extract Split Information (unchanged - use original full set) ---
    split_transactions = all_transactions_df[all_transactions_df['Type'].str.lower().isin(['split', 'stock split']) & all_transactions_df['Split Ratio'].notna() & (all_transactions_df['Split Ratio'] > 0)].sort_values(by='Date', ascending=True)
    splits_by_internal_symbol = {symbol: group[['Date', 'Split Ratio']].apply(lambda r: {'Date': r['Date'].date(), 'Split Ratio': float(r['Split Ratio'])}, axis=1).tolist() for symbol, group in split_transactions.groupby('Symbol') }

    # --- 2. Determine Required Symbols & Build Cache Keys (use EFFECTIVE tx) ---
    # --- MODIFICATION: Use transactions_df_effective here ---
    all_symbols_internal = list(set(transactions_df_effective['Symbol'].unique()))
    symbols_to_fetch_yf_portfolio = []; internal_to_yf_map = {}; yf_to_internal_map_hist = {}
    for internal_sym in all_symbols_internal:
        if internal_sym == CASH_SYMBOL_CSV: continue
        yf_sym = map_to_yf_symbol(internal_sym);
        if yf_sym: symbols_to_fetch_yf_portfolio.append(yf_sym); internal_to_yf_map[internal_sym] = yf_sym; yf_to_internal_map_hist[yf_sym] = internal_sym
    symbols_to_fetch_yf_portfolio = sorted(list(set(symbols_to_fetch_yf_portfolio)))
    symbols_for_stocks_and_benchmarks_yf = sorted(list(set(symbols_to_fetch_yf_portfolio + clean_benchmark_symbols_yf)))
    all_currencies_in_tx = set(transactions_df_effective['Local Currency'].unique()); all_currencies_needed = all_currencies_in_tx.union({display_currency})
    # --- END MODIFICATION ---
    required_fx_pairs_yf = set();
    for curr1 in all_currencies_needed:
         for curr2 in all_currencies_needed:
              if curr1 != curr2 and curr1 != 'N/A' and curr2 != 'N/A' and curr1 and curr2 and curr1.upper() != curr2.upper():
                  if curr1.upper() == 'USD' and curr2.upper() != 'USD': pair = f"USD{curr2.upper()}=X"
                  elif curr2.upper() == 'USD' and curr1.upper() != 'USD': pair = f"{curr1.upper()}USD=X"
                  else: pair = f"{curr1.upper()}{curr2.upper()}=X"
                  required_fx_pairs_yf.add(pair)
                  if curr1.upper() == 'USD' and curr2.upper() != 'USD': inv_pair = f"{curr2.upper()}USD=X"
                  elif curr2.upper() == 'USD' and curr1.upper() != 'USD': inv_pair = f"USD{curr1.upper()}=X"
                  else: inv_pair = f"{curr2.upper()}{curr1.upper()}=X"
                  required_fx_pairs_yf.add(inv_pair)
    fx_pairs_for_api_yf = sorted(list(required_fx_pairs_yf))

    # --- Generate Cache Keys (v10 - Include included_accounts_list_sorted AND excluded_accounts_list_sorted) ---
    raw_data_cache_file = f"{HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    raw_data_cache_key = f"ADJUSTED_v7::{start_date.isoformat()}::{end_date.isoformat()}::{'_'.join(sorted(symbols_for_stocks_and_benchmarks_yf))}::{'_'.join(sorted(fx_pairs_for_api_yf))}"

    daily_results_cache_file = None; daily_results_cache_key = None
    if use_daily_results_cache:
        try:
            tx_file_hash = _get_file_hash(transactions_csv_file)
            acc_map_str = json.dumps(account_currency_map, sort_keys=True)
            # --- CRITICAL: Include sorted included AND excluded accounts lists in key ---
            included_accounts_str = json.dumps(included_accounts_list_sorted) # Use sorted list
            excluded_accounts_str = json.dumps(excluded_accounts_list_sorted) # Use sorted list
            # -------------------------------------------------------------
            daily_results_cache_key = f"DAILY_RES_{CURRENT_HIST_VERSION}::{start_date.isoformat()}::{end_date.isoformat()}::{tx_file_hash}::{'_'.join(sorted(clean_benchmark_symbols_yf))}::{display_currency}::{acc_map_str}::{default_currency}::{included_accounts_str}::{excluded_accounts_str}" # <-- Added excluded_accounts_str
            cache_key_hash = hashlib.sha256(daily_results_cache_key.encode()).hexdigest()[:16]
            daily_results_cache_file = f"{DAILY_RESULTS_CACHE_PATH_PREFIX}_{cache_key_hash}.json"
            print(f"Hist ({CURRENT_HIST_VERSION}): Using DAILY results cache file: {daily_results_cache_file}")
        except Exception as e_key:
            print(f"Hist WARN ({CURRENT_HIST_VERSION}): Could not generate daily results cache key/filename: {e_key}. Daily caching disabled.")
            use_daily_results_cache = False

    # --- 3. Load or Fetch ADJUSTED Historical Raw Data (unchanged) ---
    # ... (Raw data cache logic remains the same) ...
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
    historical_fx_yf: Dict[str, pd.DataFrame] = {}
    cache_valid_raw = False
    if use_raw_data_cache and os.path.exists(raw_data_cache_file):
        try:
            with open(raw_data_cache_file, 'r') as f: cached_data = json.load(f)
            if cached_data.get('cache_key') == raw_data_cache_key:
                deserialization_errors = 0
                cached_prices = cached_data.get('historical_prices', {})
                for symbol in symbols_for_stocks_and_benchmarks_yf:
                    data_json = cached_prices.get(symbol); df = None
                    if data_json:
                         try: df = pd.read_json(StringIO(data_json), orient='split', dtype={'price': float}); df.index = pd.to_datetime(df.index).date; historical_prices_yf_adjusted[symbol] = df
                         except Exception: deserialization_errors += 1; historical_prices_yf_adjusted[symbol] = pd.DataFrame()
                    else: historical_prices_yf_adjusted[symbol] = pd.DataFrame()
                cached_fx = cached_data.get('historical_fx_rates', {})
                for pair in fx_pairs_for_api_yf:
                     data_json = cached_fx.get(pair); df = None
                     if data_json:
                          try: df = pd.read_json(StringIO(data_json), orient='split', dtype={'price': float}); df.index = pd.to_datetime(df.index).date; historical_fx_yf[pair] = df
                          except Exception: deserialization_errors += 1; historical_fx_yf[pair] = pd.DataFrame()
                     else: historical_fx_yf[pair] = pd.DataFrame()
                all_symbols_loaded = all(s in historical_prices_yf_adjusted for s in symbols_for_stocks_and_benchmarks_yf)
                all_fx_loaded = all(p in historical_fx_yf for p in fx_pairs_for_api_yf)
                if all_symbols_loaded and all_fx_loaded and deserialization_errors == 0: cache_valid_raw = True
                else: print(f"Hist WARN ({CURRENT_HIST_VERSION}): RAW cache incomplete or errors ({deserialization_errors}). Refetching."); historical_prices_yf_adjusted = {}; historical_fx_yf = {}
            else: print(f"Hist ({CURRENT_HIST_VERSION}): RAW Cache key mismatch. Refetching.")
        except Exception as e: print(f"Error reading hist RAW cache {raw_data_cache_file}: {e}. Refetching."); historical_prices_yf_adjusted = {}; historical_fx_yf = {}

    if not cache_valid_raw:
        fetched_stock_data = fetch_yf_historical(symbols_for_stocks_and_benchmarks_yf, start_date, end_date)
        fetched_fx_data = fetch_yf_historical(fx_pairs_for_api_yf, start_date, end_date)
        historical_prices_yf_adjusted = fetched_stock_data
        historical_fx_yf = fetched_fx_data
        stock_fetch_ok = bool(symbols_for_stocks_and_benchmarks_yf and historical_prices_yf_adjusted) or not symbols_for_stocks_and_benchmarks_yf
        fx_fetch_ok = bool(fx_pairs_for_api_yf and historical_fx_yf) or not fx_pairs_for_api_yf
        if use_raw_data_cache and (stock_fetch_ok or fx_fetch_ok):
            cache_content = { 'cache_key': raw_data_cache_key, 'timestamp': datetime.now().isoformat(), 'historical_prices': {symbol: df.to_json(orient='split', date_format='iso') for symbol, df in historical_prices_yf_adjusted.items()}, 'historical_fx_rates': {pair: df.to_json(orient='split', date_format='iso') for pair, df in historical_fx_yf.items()} }
            try:
                cache_dir_raw = os.path.dirname(raw_data_cache_file)
                if cache_dir_raw: os.makedirs(cache_dir_raw, exist_ok=True)
                with open(raw_data_cache_file, 'w') as f: json.dump(cache_content, f, indent=2)
            except Exception as e: print(f"Error writing hist RAW cache: {e}")
        elif not use_raw_data_cache: print(f"Hist ({CURRENT_HIST_VERSION}): RAW Caching disabled, skipping save.")
        else: print(f"Hist WARN ({CURRENT_HIST_VERSION}): Yahoo Finance RAW fetch failed or returned empty results. Cache not updated.")

    if not historical_prices_yf_adjusted and symbols_for_stocks_and_benchmarks_yf: return pd.DataFrame(), "Error: Failed to fetch/load historical stock/benchmark prices."
    if not historical_fx_yf and fx_pairs_for_api_yf: return pd.DataFrame(), "Error: Failed to fetch/load historical FX rates."
    status_msg += " Raw adjusted data loaded."

    # --- 4. Derive Unadjusted Prices (unchanged) ---
    historical_prices_yf_unadjusted = _unadjust_prices(
        historical_prices_yf_adjusted, yf_to_internal_map_hist, splits_by_internal_symbol, processed_warnings
    )
    status_msg += " Unadjusted prices derived."

    # --- Initialize daily_df for results ---
    daily_df = pd.DataFrame()
    cache_valid_daily_results = False

    # --- 5. Check Daily Results Cache (v10 - key includes included_accounts_str AND excluded_accounts_str) ---
    if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key:
        if os.path.exists(daily_results_cache_file):
            try:
                with open(daily_results_cache_file, 'r') as f:
                    cached_daily_data = json.load(f)
                if cached_daily_data.get('cache_key') == daily_results_cache_key:
                    print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Daily results cache MATCH. Loading...")
                    results_json = cached_daily_data.get('daily_results_json')
                    if results_json:
                        try:
                            daily_df = pd.read_json(StringIO(results_json), orient='split')
                            daily_df.index = pd.to_datetime(daily_df.index)
                            daily_df.sort_index(inplace=True)
                            required_cols = ['value', 'net_flow', 'daily_return', 'daily_gain']
                            for bm_symbol in clean_benchmark_symbols_yf: required_cols.append(f"{bm_symbol} Price")
                            missing_cols = [c for c in required_cols if c not in daily_df.columns]
                            if not missing_cols and not daily_df.empty:
                                print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Loaded {len(daily_df)} daily results from cache.")
                                cache_valid_daily_results = True; status_msg += " Daily results loaded from cache."
                            else: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Daily cache invalid. Recalculating."); daily_df = pd.DataFrame()
                        except Exception as e_load_df: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Error deserializing daily cache DF: {e_load_df}. Recalculating."); daily_df = pd.DataFrame()
                    else: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Daily cache missing data. Recalculating.")
                else: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Daily results cache key MISMATCH. Recalculating.")
            except Exception as e_load_cache: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Error reading daily cache: {e_load_cache}. Recalculating.")

    # --- 6. Execute Daily Calculations in Parallel (if cache invalid) ---
    if not cache_valid_daily_results:
        status_msg += " Calculating daily values..."
        # --- 6a. Determine Dates (use EFFECTIVE df) ---
        # --- MODIFICATION: Use transactions_df_effective here ---
        first_tx_date = transactions_df_effective['Date'].min().date() if not transactions_df_effective.empty else start_date
        # --- END MODIFICATION ---
        calc_start_date = max(start_date, first_tx_date)
        calc_end_date = end_date
        market_day_source_symbol = 'SPY' if 'SPY' in historical_prices_yf_adjusted else (clean_benchmark_symbols_yf[0] if clean_benchmark_symbols_yf else None)
        market_days_index = pd.Index([], dtype='object')
        if market_day_source_symbol and market_day_source_symbol in historical_prices_yf_adjusted:
            bench_df = historical_prices_yf_adjusted[market_day_source_symbol]
            if not bench_df.empty:
                 try: market_days_index = pd.Index(pd.to_datetime(bench_df.index, errors='coerce').date); market_days_index = market_days_index.dropna()
                 except Exception as e_idx: print(f"WARN: Failed converting benchmark index for market days: {e_idx}")
        if market_days_index.empty:
            print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): No market days found. Using date range."); all_dates_to_process = pd.date_range(start=calc_start_date, end=calc_end_date, freq='B').date.tolist()
        else: all_dates_to_process = market_days_index[(market_days_index >= calc_start_date) & (market_days_index <= calc_end_date)].tolist()
        if not all_dates_to_process: return pd.DataFrame(), status_msg + " No calculation dates found."

        print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Calculating {len(all_dates_to_process)} daily metrics parallel...")

        # --- 6b. Setup and Run Pool (pass EFFECTIVE df) ---
        # --- MODIFICATION: Pass transactions_df_effective to worker ---
        partial_worker = partial(
            _calculate_daily_metrics_worker,
            transactions_df=transactions_df_effective, # Pass EFFECTIVE DF
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            target_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            benchmark_symbols_yf=clean_benchmark_symbols_yf
        )
        # --- END MODIFICATION ---
        daily_results_list = []
        pool_start_time = time.time()
        if num_processes is None:
            try: num_processes = os.cpu_count();
            except NotImplementedError: num_processes = 1
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(partial_worker, all_dates_to_process, chunksize=max(1, len(all_dates_to_process) // (num_processes * 4)))
                for result in results_iterator:
                     if result: daily_results_list.append(result)
        except Exception as e_pool: print(f"Hist CRITICAL ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Pool failed: {e_pool}"); return pd.DataFrame(), status_msg + " Multiprocessing failed."
        pool_end_time = time.time()
        print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Pool finished in {pool_end_time - pool_start_time:.2f}s.")

        # --- 6c. Convert Results to DataFrame & Calculate Daily Gain/Return (unchanged) ---
        successful_results = [r for r in daily_results_list if not r.get('worker_error', False)]
        if len(successful_results) != len(all_dates_to_process): status_msg += f" ({len(all_dates_to_process) - len(successful_results)} dates failed)."
        if not successful_results: return pd.DataFrame(), status_msg + " All daily calculations failed."
        try:
            daily_df = pd.DataFrame(successful_results); daily_df['Date'] = pd.to_datetime(daily_df['Date']); daily_df.set_index('Date', inplace=True); daily_df.sort_index(inplace=True)
            daily_df['value'] = pd.to_numeric(daily_df['value'], errors='coerce'); daily_df['net_flow'] = pd.to_numeric(daily_df['net_flow'], errors='coerce')
            for bm_symbol in clean_benchmark_symbols_yf:
                col = f"{bm_symbol} Price";
                if col in daily_df.columns: daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
            initial_rows_calc = len(daily_df)
            daily_df.dropna(subset=['value'], inplace=True) # Drop rows where value calc failed
            if len(daily_df) < initial_rows_calc: status_msg += f" ({initial_rows_calc - len(daily_df)} rows dropped post-calc)."
            if daily_df.empty: return pd.DataFrame(), status_msg + " All rows dropped due to NaN portfolio value."

            previous_value = daily_df['value'].shift(1)
            daily_df['daily_gain'] = daily_df['value'] - previous_value - daily_df['net_flow'].fillna(0.0)
            daily_df['daily_return'] = np.nan
            valid_prev_value_mask = previous_value.notna() & (abs(previous_value) > 1e-9)
            daily_df.loc[valid_prev_value_mask, 'daily_return'] = daily_df.loc[valid_prev_value_mask, 'daily_gain'] / previous_value.loc[valid_prev_value_mask]
            zero_gain_mask = abs(daily_df['daily_gain']) < 1e-9
            zero_prev_value_mask = previous_value.notna() & (abs(previous_value) <= 1e-9)
            daily_df.loc[zero_gain_mask & zero_prev_value_mask, 'daily_return'] = 0.0
            if not daily_df.empty:
                 daily_df.iloc[0, daily_df.columns.get_loc('daily_gain')] = np.nan
                 daily_df.iloc[0, daily_df.columns.get_loc('daily_return')] = np.nan
            if 'daily_gain' not in daily_df.columns or 'daily_return' not in daily_df.columns: return pd.DataFrame(), status_msg + " Failed calc daily gain/return."
            status_msg += f" {len(daily_df)} days calculated."

        except Exception as e_df_create: print(f"Hist CRITICAL ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Failed create/process DF: {e_df_create}"); return pd.DataFrame(), status_msg + " Error processing results."

        # --- 6d. Save Enhanced Daily Results Cache (v10 - key includes included_accounts_str AND excluded_accounts_str) ---
        if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key and not daily_df.empty:
            print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Saving daily results to cache: {daily_results_cache_file}")
            cache_content = { 'cache_key': daily_results_cache_key, 'timestamp': datetime.now().isoformat(), 'daily_results_json': daily_df.to_json(orient='split', date_format='iso') }
            try:
                cache_dir = os.path.dirname(daily_results_cache_file);
                if cache_dir: os.makedirs(cache_dir, exist_ok=True);
                with open(daily_results_cache_file, 'w') as f: json.dump(cache_content, f, indent=2)
            except Exception as e_save_cache: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Error writing daily cache: {e_save_cache}")

    # --- 7. Process Results and Calculate Accumulated Gains (unchanged) ---
    # ... (Accumulated gain calculation logic remains the same) ...
    if daily_df.empty: return pd.DataFrame(), status_msg + " No daily data generated."
    if not all(col in daily_df.columns for col in ['value', 'daily_return', 'daily_gain']): return pd.DataFrame(), status_msg + " Data prep error before final calcs."

    print(f"Hist ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Calculating accumulated gains...")
    try:
        # Portfolio Accumulated Gain (Based on the filtered scope)
        gain_factors_portfolio = (1 + daily_df['daily_return'].fillna(0.0))
        daily_df['Portfolio Accumulated Gain'] = gain_factors_portfolio.cumprod()
        daily_df.loc[daily_df['daily_return'].isna(), 'Portfolio Accumulated Gain'] = np.nan
        if not daily_df.empty and 'Portfolio Accumulated Gain' in daily_df.columns:
             final_twr_factor = daily_df['Portfolio Accumulated Gain'].iloc[-1] # Get the last factor

        # Benchmark Accumulated Gain (Remains the same)
        for bm_symbol in clean_benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"; accum_col = f"{bm_symbol} Accumulated Gain"
            if price_col in daily_df.columns:
                bench_prices_no_na = daily_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    bench_daily_returns = bench_prices_no_na.pct_change().fillna(0.0)
                    gain_factors_bench = (1 + bench_daily_returns)
                    accum_gains_bench = gain_factors_bench.cumprod()
                    daily_df[accum_col] = accum_gains_bench.reindex(daily_df.index, method='ffill')
                    if not daily_df.empty and accum_col in daily_df.columns:
                         first_valid_price_idx = bench_prices_no_na.index.min()
                         if first_valid_price_idx in daily_df.index: daily_df.loc[first_valid_price_idx, accum_col] = np.nan
                else: daily_df[accum_col] = np.nan
            else: daily_df[accum_col] = np.nan
        status_msg += " Accum gain calc complete."

    except Exception as e_seq: print(f"Hist CRITICAL ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Accum gain calc error: {e_seq}"); return pd.DataFrame(), status_msg + " Accum gain calc failed."

    # --- 8. Select Final Columns & Apply Interval (unchanged) ---
    # ... (Column selection and resampling logic remains the same) ...
    columns_to_keep = ['value', 'daily_gain', 'daily_return', 'Portfolio Accumulated Gain']
    for bm_symbol in clean_benchmark_symbols_yf:
        if f"{bm_symbol} Price" in daily_df.columns: columns_to_keep.append(f"{bm_symbol} Price")
        if f"{bm_symbol} Accumulated Gain" in daily_df.columns: columns_to_keep.append(f"{bm_symbol} Accumulated Gain")
    columns_to_keep = [col for col in columns_to_keep if col in daily_df.columns]
    final_df_filtered = daily_df[columns_to_keep].copy()
    final_df_filtered.rename(columns={'value': 'Portfolio Value', 'daily_gain': 'Portfolio Daily Gain'}, inplace=True)

    if interval != 'D':
        try:
            resample_freq = interval
            resampling_agg = { 'Portfolio Value': 'last', 'Portfolio Daily Gain': 'sum' }
            for bm_symbol in clean_benchmark_symbols_yf:
                 if f"{bm_symbol} Price" in final_df_filtered.columns: resampling_agg[f"{bm_symbol} Price"] = 'last'
            cols_to_drop_resample = ['daily_return', 'Portfolio Accumulated Gain']
            for bm in clean_benchmark_symbols_yf:
                if f"{bm} Accumulated Gain" in final_df_filtered.columns: cols_to_drop_resample.append(f"{bm} Accumulated Gain")
            cols_to_drop_resample = [c for c in cols_to_drop_resample if c in final_df_filtered.columns]
            final_df_resampled = final_df_filtered.drop(columns=cols_to_drop_resample, errors='ignore').resample(resample_freq).agg(resampling_agg)

            if not final_df_resampled.empty:
                if 'Portfolio Value' in final_df_resampled.columns:
                    resampled_returns = final_df_resampled['Portfolio Value'].pct_change()
                    if not resampled_returns.empty: resampled_returns.iloc[0] = np.nan
                    final_df_resampled['Portfolio Accumulated Gain'] = (1 + resampled_returns.fillna(0.0)).cumprod()
                    if not resampled_returns.empty and pd.isna(resampled_returns.iloc[0]) and 'Portfolio Accumulated Gain' in final_df_resampled.columns:
                        final_df_resampled.iloc[0, final_df_resampled.columns.get_loc('Portfolio Accumulated Gain')] = np.nan
                    if 'Portfolio Accumulated Gain' in final_df_resampled.columns:
                        final_twr_factor = final_df_resampled['Portfolio Accumulated Gain'].iloc[-1]
                else: final_df_resampled['Portfolio Accumulated Gain'] = np.nan

                for bm_symbol in clean_benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"; accum_col = f"{bm_symbol} Accumulated Gain"
                    if price_col in final_df_resampled.columns:
                        resampled_bench_returns = final_df_resampled[price_col].pct_change()
                        if not resampled_bench_returns.empty: resampled_bench_returns.iloc[0] = np.nan
                        final_df_resampled[accum_col] = (1 + resampled_bench_returns.fillna(0.0)).cumprod()
                        if not resampled_bench_returns.empty and pd.isna(resampled_bench_returns.iloc[0]) and accum_col in final_df_resampled.columns:
                             final_df_resampled.iloc[0, final_df_resampled.columns.get_loc(accum_col)] = np.nan
                    else: final_df_resampled[accum_col] = np.nan
                final_df_filtered = final_df_resampled
        except Exception as e_resample: print(f"Hist WARN ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Failed resampling: {e_resample}. Returning daily.")

    # --- 9. Return Results ---
    end_time_hist = time.time()
    print(f"--- Historical Performance Calculation Finished ({CURRENT_HIST_VERSION} / Scope: {filter_desc}) ---")
    print(f"Total Historical Calc Time: {end_time_hist - start_time_hist:.2f} seconds")

    # --- Include final TWR factor in status message (unchanged) ---
    final_status = status_msg
    if 'failed' in status_msg.lower() or 'error' in status_msg.lower(): pass
    elif final_df_filtered.empty: final_status += " Success (Empty Result)."
    else: final_status += " Success."
    final_status += f"|||TWR_FACTOR:{final_twr_factor:.6f}" if pd.notna(final_twr_factor) else "|||TWR_FACTOR:NaN"

    # --- Final column ordering (unchanged) ---
    final_cols_order = ['Portfolio Value', 'Portfolio Daily Gain', 'daily_return', 'Portfolio Accumulated Gain']
    for bm in clean_benchmark_symbols_yf:
        if f"{bm} Price" in final_df_filtered.columns: final_cols_order.append(f"{bm} Price")
        if f"{bm} Accumulated Gain" in final_df_filtered.columns: final_cols_order.append(f"{bm} Accumulated Gain")
    final_cols_order = [c for c in final_cols_order if c in final_df_filtered.columns]
    final_df_filtered = final_df_filtered[final_cols_order]

    return final_df_filtered, final_status

# --- Example Usage (Main block for testing this file directly) ---
if __name__ == '__main__':
    print("\nRunning portfolio_logic.py as main script for testing (v10 - Account/Currency)...")
    # --- Configuration for Test Run ---
    test_csv_file = 'my_transactions.csv' # Needs to exist
    test_display_currency = 'EUR'
    # --- Define Test Account Currency Map ---
    test_account_currency_map = {
        'SET': 'THB',
        'IBKR': 'USD',
        'Fidelity': 'USD',
        'E*TRADE':'USD',
        'TD Ameritrade': 'USD',
        'Sharebuilder': 'USD',
        'Unknown': 'USD', # Specify currency for 'Unknown' if it exists
        'Dime!': 'USD'    # Specify currency for 'Dime!'
    }
    test_default_currency = 'USD' # Define the default fallback

    # --- Test Historical Performance Calculation ---
    print("\n--- Testing Historical Performance Calculation (v10 - Account/Currency) ---")
    test_start = date(2023, 1, 1); test_end = date(2024, 6, 30); test_interval = 'M'; test_benchmarks = ['SPY', 'QQQ']
    test_use_raw_cache_flag = True; test_use_daily_results_cache_flag = True; test_num_processes = None
    test_accounts_all = None; test_accounts_subset1 = ['SET', 'IBKR']; test_exclude_set = ['SET']

    test_scenarios = {
        # "All": {"include": test_accounts_all, "exclude": None},
        # "All_Exclude_SET": {"include": test_accounts_all, "exclude": test_exclude_set},
        "SET_Only": {"include": ['SET'], "exclude": None},
        # "IBKR_E*TRADE": {"include": test_accounts_subset1, "exclude": None},
    }

    if not os.path.exists(test_csv_file):
         print(f"ERROR: Test transactions file '{test_csv_file}' not found. Cannot run tests.")
    else:
        for name, scenario in test_scenarios.items():
            print(f"\n--- Running Historical Test: Scenario='{name}', Include={scenario['include']}, Exclude={scenario['exclude']} ---")
            start_time_run = time.time()
            hist_df, hist_status = calculate_historical_performance(
                transactions_csv_file=test_csv_file,
                start_date=test_start,
                end_date=test_end,
                interval=test_interval,
                benchmark_symbols_yf=test_benchmarks,
                display_currency=test_display_currency,
                account_currency_map=test_account_currency_map, # <-- Pass map
                default_currency=test_default_currency,          # <-- Pass default
                use_raw_data_cache=test_use_raw_cache_flag,
                use_daily_results_cache=test_use_daily_results_cache_flag,
                num_processes=test_num_processes,
                include_accounts=scenario['include'],
                exclude_accounts=scenario['exclude']
            )
            end_time_run = time.time()
            print(f"Test '{name}' Status: {hist_status}")
            print(f"Test '{name}' Execution Time: {end_time_run - start_time_run:.2f} seconds")
            if not hist_df.empty: print(f"Test '{name}' DF tail:\n{hist_df.tail().to_string()}")
            else: print(f"Test '{name}' Result: Empty DataFrame")

        # --- Test Current Portfolio Summary ---
        print("\n--- Testing Current Portfolio Summary (Account Inclusion/Currency) ---")
        summary_metrics, holdings_df, ignored_df, account_metrics, status = calculate_portfolio_summary(
            transactions_csv_file=test_csv_file,
            display_currency=test_display_currency,
            account_currency_map=test_account_currency_map, # <-- Pass map
            default_currency=test_default_currency,          # <-- Pass default
            include_accounts=test_accounts_subset1 # Test with a subset
        )
        print(f"Current Summary Status (Subset {test_accounts_subset1}): {status}")
        if summary_metrics: print(f"Overall Metrics (Subset): {summary_metrics}")
        if holdings_df is not None and not holdings_df.empty: print(f"Holdings DF (Subset) Head:\n{holdings_df.head().to_string()}")
        if account_metrics: print(f"Account Metrics (Subset): {account_metrics}")

# Example test cases (add inside if __name__ == '__main__':)
print("\n--- IRR/NPV Test Cases ---")
dates1 = [date(2023, 1, 1), date(2024, 1, 1)]
flows1 = [-100, 110] # Simple 10% return
irr1 = calculate_irr(dates1, flows1)
print(f"Test 1: Flows={flows1}, IRR={irr1*100 if irr1 is not np.nan else 'NaN':.2f}% (Expected ~10%)")
npv1_at_10pct = calculate_npv(0.10, dates1, flows1)
print(f"Test 1: NPV @ 10% = {npv1_at_10pct:.2f} (Expected ~0.00)")
npv1_at_5pct = calculate_npv(0.05, dates1, flows1)
print(f"Test 1: NPV @ 5% = {npv1_at_5pct:.2f}")

dates2 = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]
flows2 = [-100, -50, 170] # Multiple flows
irr2 = calculate_irr(dates2, flows2)
print(f"Test 2: Flows={flows2}, IRR={irr2*100 if irr2 is not np.nan else 'NaN':.2f}%")
if pd.notna(irr2):
     npv2_at_irr = calculate_npv(irr2, dates2, flows2)
     print(f"Test 2: NPV @ {irr2*100:.2f}% = {npv2_at_irr:.4f} (Expected ~0.00)")

dates3 = [date(2023,1,1), date(2024,1,1)]
flows3 = [-100, 90] # Negative return
irr3 = calculate_irr(dates3, flows3)
print(f"Test 3: Flows={flows3}, IRR={irr3*100 if irr3 is not np.nan else 'NaN':.2f}% (Expected ~-10%)")

dates4 = [date(2023,1,1), date(2024,1,1)]
flows4 = [100, -110] # Invalid for standard IRR (all positive/negative after sign flip)
irr4 = calculate_irr(dates4, flows4)
print(f"Test 4: Flows={flows4}, IRR={irr4*100 if irr4 is not np.nan else 'NaN'} (Expected NaN)")
# --- END IRR/NPV Test Cases ---

print("\n--- Running SIMPLE Historical Test ---")
# test_csv_file = 'test_hist_simple.csv' # Point to the simple test file
test_csv_file = 'test_hist_simple.csv' # Point to the simple test file
test_start = date(2023, 1, 1)
test_end = date(2023, 7, 1) # Choose a relevant end date
test_interval = 'D' # Use Daily interval for detailed checking
test_display_currency = 'USD' # Or 'THB'
test_account_currency_map = {'ACC_USD': 'USD', 'ACC_THB': 'THB'}
test_default_currency = 'USD'
test_benchmarks = [] # No benchmarks needed for this specific test
test_use_raw_cache_flag = False # Force fetch
test_use_daily_results_cache_flag = False # Force calc
test_num_processes = 1 # Easier to debug single process first
test_include_accounts = None # Include both accounts
test_exclude_accounts = None
if not os.path.exists(test_csv_file):
    print(f"ERROR: Test file '{test_csv_file}' not found.")
else:
    hist_df, hist_status = calculate_historical_performance(
        transactions_csv_file=test_csv_file,
        start_date=test_start,
        end_date=test_end,
        interval=test_interval,
        benchmark_symbols_yf=test_benchmarks,
        display_currency=test_display_currency,
        account_currency_map=test_account_currency_map,
        default_currency=test_default_currency,
        use_raw_data_cache=test_use_raw_cache_flag,
        use_daily_results_cache=test_use_daily_results_cache_flag,
        num_processes=test_num_processes,
        include_accounts=test_include_accounts,
        exclude_accounts=test_exclude_accounts
        )
    print(f"\nSimple Test Status: {hist_status}")
    if not hist_df.empty:
        print("\nSimple Test DataFrame (Selected Dates):")
        # Select specific dates around events to check
        dates_to_check = [
            date(2023, 1, 10), # After AAPL buy
            date(2023, 1, 16), # After BEM buy
            date(2023, 2, 1),  # Day of USD Deposit
            date(2023, 2, 2),  # Day after Deposit
            date(2023, 3, 10), # Day of AAPL Dividend (check if flow reflects?) -> No, cash flow doesn't include divs
            date(2023, 4, 3),  # Day of AAPL Split (check value calc reflects split)
            date(2023, 4, 4),  # Day after Split
            date(2023, 5, 10), # Day of AAPL Sell
            date(2023, 5, 15), # Day of BEM Sell
            date(2023, 6, 1),  # Day of USD Withdrawal
            date(2023, 6, 2),  # Day after Withdrawal
            date(2023, 6, 30)  # Near end
            ]
            # Convert dates to Timestamps for indexing if needed
        ts_to_check = [pd.Timestamp(d) for d in dates_to_check if pd.Timestamp(d) in hist_df.index]
        print(hist_df.loc[ts_to_check].to_string())

        # Optional: Verify specific values manually/programmatically
        # e.g., check value on 2023-04-04 vs 2023-04-03 considering AAPL split
        # e.g., check net_flow on 2023-02-01 (+500) and 2023-06-01 (-100)
    else:
        print("Simple Test Result: Empty DataFrame")
# --- END OF FILE portfolio_logic.py ---
