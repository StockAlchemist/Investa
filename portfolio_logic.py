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
from collections import defaultdict # Ensure defaultdict is imported

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

def _process_transactions_to_holdings(
    transactions_df: pd.DataFrame,      # The filtered DataFrame to process
    default_currency: str,              # Needed for initializing holding currency if missing? (Shouldn't happen now)
    shortable_symbols: Set[str]         # Set of symbols allowed for shorting
    # Do we need ignored_indices/reasons here? Yes, modify them by reference or return them. Let's return them.
) -> Tuple[Dict[Tuple[str, str], Dict], Dict[str, float], Dict[str, float], Dict[str, float], Set[int], Dict[int, str]]:
    """
    Processes stock/ETF transactions (excluding $CASH) to calculate holdings,
    realized gains, dividends, and commissions in their local currencies.

    Args:
        transactions_df: DataFrame of transactions filtered for the desired scope.
                         MUST include 'Local Currency' column.
        default_currency: Default currency string.
        shortable_symbols: Set of symbols allowed for shorting.


    Returns:
        A tuple containing:
        - holdings: Dict keyed by (symbol, account) with holding details (qty, costs, gains etc. in local currency).
        - overall_realized_gains_local: Dict keyed by currency code summing realized gains.
        - overall_dividends_local: Dict keyed by currency code summing dividends.
        - overall_commissions_local: Dict keyed by currency code summing commissions.
        - ignored_row_indices: Set of original_index values for rows ignored during THIS processing step.
        - ignored_reasons: Dict mapping original_index to reason for ignoring in THIS step.
    """
    holdings: Dict[Tuple[str, str], Dict] = {}
    overall_realized_gains_local: Dict[str, float] = defaultdict(float)
    overall_dividends_local: Dict[str, float] = defaultdict(float)
    overall_commissions_local: Dict[str, float] = defaultdict(float)
    ignored_row_indices_local = set() # Use local set for this function's scope
    ignored_reasons_local = {}       # Use local dict

    # --- Start of code to MOVE from calculate_portfolio_summary ---
    print("Processing filtered stock/ETF transactions...") # Modify print message

    for index, row in transactions_df.iterrows():
        # Skip $CASH transactions here
        if row['Symbol'] == CASH_SYMBOL_CSV:
             continue

        original_index = row['original_index'] # Make sure this column exists from _load_and_clean...
        symbol = row['Symbol']
        # No need to check for CASH again here
        account, tx_type = row['Account'], row['Type']
        qty, price_local, total_amount_local = row['Quantity'], row['Price/Share'], row['Total Amount']
        commission_local, split_ratio = row['Commission'], row['Split Ratio']
        # --- Use Local Currency from DataFrame ---
        local_currency = row['Local Currency'] # Assumes this column exists!
        tx_date = row['Date'].date() # Added tx_date, might be useful for debugging later
        # ---------------------------------------

        holding_key = (symbol, account)

        # Initialize holding if first time seeing this symbol/account combo
        if holding_key not in holdings:
            holdings[holding_key] = {
                'qty': 0.0,
                'total_cost_local': 0.0,
                'realized_gain_local': 0.0,
                'dividends_local': 0.0,
                'commissions_local': 0.0,
                'local_currency': local_currency, # Store the currency
                'short_proceeds_local': 0.0,      # For short positions
                'short_original_qty': 0.0,        # For short positions avg cost
                'total_cost_invested_local': 0.0, # Tracks invested cost basis over time
                'cumulative_investment_local': 0.0 # Sum of buys/covers + fees - sell/short proceeds
            }
        # Safety check (shouldn't happen if _load_and_clean works)
        elif holdings[holding_key]['local_currency'] != local_currency:
            msg=f"Currency mismatch for {symbol}/{account}";
            print(f"CRITICAL WARN in _process_transactions: {msg} row {original_index}. Skip.");
            ignored_reasons_local[original_index] = msg
            ignored_row_indices_local.add(original_index)
            continue

        holding = holdings[holding_key]
        commission_for_overall = commission_local # Assume commission applies unless skipped

        # --- Core Transaction Processing Logic ---
        try:
            # Shorting Logic
            if symbol in shortable_symbols and tx_type in ['short sell', 'buy to cover']:
                qty_abs = abs(qty);
                if tx_type == 'short sell':
                    if qty_abs <= 1e-9: raise ValueError("Short Sell qty must be > 0")
                    proceeds = (qty_abs * price_local) - commission_local;
                    holding['qty'] -= qty_abs;
                    holding['short_proceeds_local'] += proceeds;
                    holding['short_original_qty'] += qty_abs;
                    holding['commissions_local'] += commission_local
                    # Note: cumulative_investment decreases on short sell (like a sell)
                    holding['cumulative_investment_local'] -= proceeds # Proceeds received reduce net investment
                elif tx_type == 'buy to cover':
                    if qty_abs <= 1e-9: raise ValueError("Buy Cover qty must be > 0")
                    qty_currently_short = abs(holding['qty']) if holding['qty'] < -1e-9 else 0.0;
                    if qty_currently_short < 1e-9: raise ValueError(f"Not currently short {symbol}/{account} to cover.")
                    qty_covered = min(qty_abs, qty_currently_short);
                    cost = (qty_covered * price_local) + commission_local
                    if holding['short_original_qty'] <= 1e-9: raise ZeroDivisionError(f"Short original qty is zero/neg for {symbol}/{account}")
                    avg_proceeds_per_share = holding['short_proceeds_local'] / holding['short_original_qty'];
                    proceeds_attributed = qty_covered * avg_proceeds_per_share;
                    gain = proceeds_attributed - cost # Gain = Proceeds - Cost to Cover
                    holding['qty'] += qty_covered;
                    holding['short_proceeds_local'] -= proceeds_attributed;
                    holding['short_original_qty'] -= qty_covered
                    holding['commissions_local'] += commission_local;
                    holding['realized_gain_local'] += gain;
                    overall_realized_gains_local[local_currency] += gain;
                    # total_cost_invested_local shouldn't decrease here, cost_basis handled by qty change
                    # Reset short tracking if fully covered
                    if abs(holding['short_original_qty']) < 1e-9: holding['short_proceeds_local'] = 0.0; holding['short_original_qty'] = 0.0
                    if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0; # Should become zero after covering
                    holding['cumulative_investment_local'] += cost # Cost to cover increases net investment
                continue # Skip rest of logic for short types

            # Standard Buy/Sell/Dividend/Fee/Split
            if tx_type == 'buy' or tx_type == 'deposit': # Treat deposit of stock like a buy at price
                qty_abs = abs(qty);
                if qty_abs <= 1e-9: raise ValueError("Buy/Deposit qty must be > 0")
                cost = (qty_abs * price_local) + commission_local;
                holding['qty'] += qty_abs;
                holding['total_cost_local'] += cost;
                holding['commissions_local'] += commission_local;
                holding['total_cost_invested_local'] += cost; # Add to invested cost
                holding['cumulative_investment_local'] += cost # Buy increases net investment

            elif tx_type == 'sell' or tx_type == 'withdrawal': # Treat withdrawal like a sell
                qty_abs = abs(qty);
                held_qty = holding['qty'];
                # Can only sell long positions with this logic
                if held_qty <= 1e-9:
                    msg=f"Sell attempt {symbol}/{account} w/ non-positive long qty ({held_qty:.4f})";
                    print(f"Warn in _process_transactions: {msg} row {original_index}. Skip.");
                    ignored_reasons_local[original_index]=msg; ignored_row_indices_local.add(original_index);
                    continue # Skip if not holding long shares
                if qty_abs <= 1e-9: raise ValueError("Sell/Withdrawal qty must be > 0")

                qty_sold = min(qty_abs, held_qty);
                cost_sold = 0.0
                # Calculate cost basis of shares sold (pro-rata)
                if held_qty > 1e-9 and abs(holding['total_cost_local']) > 1e-9:
                    cost_sold = qty_sold * (holding['total_cost_local'] / held_qty)

                proceeds = (qty_sold * price_local) - commission_local;
                gain = proceeds - cost_sold # Realized Gain = Proceeds - Cost Basis Sold

                holding['qty'] -= qty_sold;
                holding['total_cost_local'] -= cost_sold; # Reduce cost basis
                holding['commissions_local'] += commission_local;
                holding['realized_gain_local'] += gain;
                overall_realized_gains_local[local_currency] += gain;
                holding['total_cost_invested_local'] -= cost_sold # Reduce cost basis for invested tracking too
                # Reset cost basis to 0 if quantity becomes 0
                if abs(holding['qty']) < 1e-9:
                    holding['qty'] = 0.0;
                    holding['total_cost_local'] = 0.0;
                # Sell reduces net investment
                holding['cumulative_investment_local'] -= proceeds # Proceeds received reduce net investment

            elif tx_type == 'dividend':
                div_amt_local = 0.0;
                qty_abs = abs(qty) if pd.notna(qty) else 0
                # Determine dividend amount (prefer Total Amount if provided)
                if pd.notna(total_amount_local) and total_amount_local != 0:
                    div_amt_local = total_amount_local
                elif pd.notna(price_local) and price_local != 0: # Use price/share if total amount missing
                    # If quantity is given, use qty*price, otherwise assume price IS the total dividend
                    div_amt_local = (qty_abs * price_local) if qty_abs > 0 else price_local

                # Dividend effect depends on whether position is long or short
                # Assume dividend received for long, paid for short (may need adjustment based on broker)
                div_effect = abs(div_amt_local) if (holding['qty'] >= -1e-9 or symbol not in shortable_symbols) else -abs(div_amt_local)
                holding['dividends_local'] += div_effect;
                overall_dividends_local[local_currency] += div_effect;
                holding['commissions_local'] += commission_local # Fees associated with dividend
                # Dividends are returns, reduce net investment
                holding['cumulative_investment_local'] -= div_effect # Dividend received reduces net investment

            elif tx_type == 'fees':
                 fee_cost = abs(commission_local);
                 holding['commissions_local'] += fee_cost;
                 holding['total_cost_invested_local'] += fee_cost; # Fees add to cost basis? Debatable, but consistent here.
                 holding['cumulative_investment_local'] += fee_cost # Fees increase net investment

            elif tx_type in ['split', 'stock split']:
                if pd.isna(split_ratio) or split_ratio <= 0:
                    raise ValueError(f"Invalid split ratio: {split_ratio}")
                old_qty = holding['qty']
                if abs(old_qty) >= 1e-9: # Apply split only if holding shares
                    holding['qty'] *= split_ratio
                    # Adjust short original quantity if shorting
                    if old_qty < -1e-9 and symbol in shortable_symbols:
                         holding['short_original_qty'] *= split_ratio
                    # Clean up potential near-zero values after split
                    if abs(holding['qty']) < 1e-9: holding['qty'] = 0.0
                    if abs(holding['short_original_qty']) < 1e-9: holding['short_original_qty'] = 0.0
                # Cost basis ('total_cost_local') remains the same during a split
                holding['commissions_local'] += commission_local; # Add any fees associated with split
                holding['total_cost_invested_local'] += commission_local; # Track fees
                holding['cumulative_investment_local'] += commission_local # Fees increase net investment

            else:
                # Handle unrecognized transaction types for stocks/ETFs
                msg=f"Unhandled stock tx type '{tx_type}'";
                print(f"Warn in _process_transactions: {msg} row {original_index}. Skip.");
                ignored_reasons_local[original_index]=msg; ignored_row_indices_local.add(original_index);
                commission_for_overall = 0.0; # Don't add commission if tx is skipped
                continue # Skip to next transaction

            # Add commission to overall total if the transaction wasn't skipped
            if commission_for_overall != 0:
                overall_commissions_local[local_currency] += abs(commission_for_overall)

        except (ValueError, TypeError, ZeroDivisionError, KeyError, Exception) as e:
            error_msg = f"Processing Error: {e}";
            print(f"ERROR in _process_transactions processing row {original_index} ({symbol}, {tx_type}): {e}. Skipping row.");
            # Optionally add traceback print here for debugging:
            # import traceback
            # traceback.print_exc()
            ignored_reasons_local[original_index] = error_msg;
            ignored_row_indices_local.add(original_index);
            continue # Skip to next transaction

    # --- End of code to MOVE ---

    return (
        holdings,
        dict(overall_realized_gains_local), # Convert defaultdict to dict
        dict(overall_dividends_local),      # Convert defaultdict to dict
        dict(overall_commissions_local),     # Convert defaultdict to dict
        ignored_row_indices_local,
        ignored_reasons_local
    )

def _calculate_cash_balances(
    transactions_df: pd.DataFrame, # The filtered DataFrame
    default_currency: str         # Fallback currency
    # No ignored sets needed here as $CASH txns are usually simpler
) -> Dict[str, Dict]:
    """
    Calculates final cash balances, dividends, and commissions for $CASH
    symbol transactions within each account, using their local currency.

    Args:
        transactions_df: DataFrame of transactions filtered for the desired scope.
                         MUST include 'Local Currency' column.
        default_currency: Default currency string.

    Returns:
        Dict keyed by account name containing $CASH details:
        {'qty': balance, 'dividends': divs, 'commissions': fees, 'currency': local_curr}
    """
    cash_summary: Dict[str, Dict] = {}

    # --- Start of code to MOVE ---
    try:
        # Filter only $CASH transactions from the input DataFrame
        cash_transactions = transactions_df[transactions_df['Symbol'] == CASH_SYMBOL_CSV].copy()

        if not cash_transactions.empty:
            # Calculate signed quantity for deposits/withdrawals
            def get_signed_quantity_cash(row):
                # Ensure 'Type' and 'Quantity' columns exist and handle potential errors
                type_lower = str(row.get('Type', '')).lower()
                qty_val = row.get('Quantity')
                qty = pd.to_numeric(qty_val, errors='coerce')
                # Return 0 if quantity is NaN
                if pd.isna(qty): return 0.0
                # Handle transaction types
                if type_lower in ['buy', 'deposit']: return abs(qty)
                elif type_lower in ['sell', 'withdrawal']: return -abs(qty)
                else: return 0.0 # Ignore other types for balance calc

            cash_transactions['SignedQuantity'] = cash_transactions.apply(get_signed_quantity_cash, axis=1)

            # Aggregate quantities and commissions by account
            cash_qty_agg = cash_transactions.groupby('Account')['SignedQuantity'].sum()
            # Fill NaN commissions with 0 before summing
            cash_comm_agg = cash_transactions.groupby('Account')['Commission'].fillna(0.0).sum()


            # Calculate net dividends specifically for $CASH transactions
            cash_dividends_tx = cash_transactions[cash_transactions['Type'] == 'dividend'].copy()
            cash_div_agg = pd.Series(dtype=float)
            if not cash_dividends_tx.empty:
                # Calculate dividend amount (prefer 'Total Amount', fallback to 'Price/Share' or qty*price)
                def get_dividend_amount(r):
                    total_amt = pd.to_numeric(r.get('Total Amount'), errors='coerce')
                    price = pd.to_numeric(r.get('Price/Share'), errors='coerce')
                    qty = pd.to_numeric(r.get('Quantity'), errors='coerce')
                    qty_abs = abs(qty) if pd.notna(qty) else 0.0

                    if pd.notna(total_amt) and total_amt != 0: return total_amt
                    elif pd.notna(price) and price != 0:
                        # If quantity is meaningful, use qty*price, otherwise assume price is the amount
                        return (qty_abs * price) if qty_abs > 0 else price
                    else: return 0.0

                cash_dividends_tx['DividendAmount'] = cash_dividends_tx.apply(get_dividend_amount, axis=1)
                cash_dividends_tx['Commission'] = cash_dividends_tx['Commission'].fillna(0.0) # Ensure commission is numeric
                cash_dividends_tx['NetDividend'] = cash_dividends_tx['DividendAmount'] - cash_dividends_tx['Commission']
                cash_div_agg = cash_dividends_tx.groupby('Account')['NetDividend'].sum()

            # Get the local currency for each account from the $CASH transactions
            # Use .first() assuming currency is consistent per account within $CASH txns
            cash_currency_map = cash_transactions.groupby('Account')['Local Currency'].first()

            # Combine aggregations
            all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index).union(cash_comm_agg.index).union(cash_div_agg.index)

            for acc in all_cash_accounts:
                # Use default_currency if somehow missing from the filtered cash transactions
                acc_currency = cash_currency_map.get(acc, default_currency)
                acc_balance = cash_qty_agg.get(acc, 0.0)
                acc_commissions = cash_comm_agg.get(acc, 0.0)
                acc_dividends_only = cash_div_agg.get(acc, 0.0)

                cash_summary[acc] = {
                    'qty': acc_balance,
                    'realized': 0.0, # Realized gains aren't typically tracked for cash itself
                    'dividends': acc_dividends_only,
                    'commissions': acc_commissions,
                    'currency': acc_currency # Store the determined local currency
                }
        else:
            print("Info: No $CASH transactions found in the filtered set for balance calculation.")
    except Exception as e:
        print(f"ERROR calculating $CASH balances separately: {e}");
        import traceback
        traceback.print_exc();
        cash_summary = {} # Return empty on error
    # --- End of code to MOVE ---

    return cash_summary

def _build_summary_rows( # Renamed from _build_summary_dataframe for clarity
    holdings: Dict[Tuple[str, str], Dict],
    cash_summary: Dict[str, Dict],
    current_stock_data: Dict[str, Dict[str, Optional[float]]], # Price, change etc. keyed by internal symbol
    current_fx_rates_vs_usd: Dict[str, float], # CURR per USD rates
    display_currency: str,
    default_currency: str,
    transactions_df: pd.DataFrame, # Needed for IRR calculation fallback/lookup
    report_date: date,
    shortable_symbols: Set[str], # Needed for unrealized gain calc for shorts
    excluded_symbols: Set[str] # Needed for price source fallback logic
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, str]]:
    """
    Calculates final display values for each holding (stocks and cash),
    including market value, gains, IRR, etc., in the specified display_currency.

    Args:
        holdings: Processed stock/ETF holdings keyed by (symbol, account).
        cash_summary: Processed cash balances keyed by account.
        current_stock_data: Dictionary with current price/change data from fetcher.
        current_fx_rates_vs_usd: Dictionary with CURR per USD rates.
        display_currency: The target currency for output values.
        default_currency: Default currency for fallbacks.
        transactions_df: Filtered transaction DataFrame for IRR calculation.
        report_date: The date for which the summary is being generated.
        shortable_symbols: Set of symbols allowed for shorting.
        excluded_symbols: Set of symbols excluded from price fetching.

    Returns:
        A tuple containing:
        - portfolio_summary_rows: A list of dictionaries, each representing a row for the final summary DataFrame.
        - account_market_values_local: Dict keyed by account summing market value in LOCAL currency.
        - account_local_currency_map: Dict mapping account to its determined local currency.
    """
    portfolio_summary_rows: List[Dict[str, Any]] = [] # Changed name for clarity
    account_market_values_local: Dict[str, float] = defaultdict(float)
    account_local_currency_map: Dict[str, str] = {}

    print(f"Calculating final portfolio summary rows in {display_currency}...") # Update print

    # --- Loop 1: Process Stock/ETF Holdings ---
    for holding_key, data in holdings.items():
        # --- Start of code to MOVE (Stock Holding Loop) ---
        symbol, account = holding_key
        # Get holding details (local currency)
        current_qty = data.get('qty', 0.0)
        realized_gain_local = data.get('realized_gain_local', 0.0)
        dividends_local = data.get('dividends_local', 0.0)
        commissions_local = data.get('commissions_local', 0.0)
        local_currency = data.get('local_currency', default_currency)
        current_total_cost_local = data.get('total_cost_local', 0.0)
        short_proceeds_local = data.get('short_proceeds_local', 0.0)
        # short_original_qty = data.get('short_original_qty', 0.0) # Needed for short gain calc below
        total_cost_invested_local = data.get('total_cost_invested_local', 0.0)
        cumulative_investment_local = data.get('cumulative_investment_local', 0.0)

        # Store account currency mapping
        account_local_currency_map[account] = local_currency

        # Get current market data for the symbol
        stock_data = current_stock_data.get(symbol, {})
        current_price_local_raw = stock_data.get('price') # Price is in the stock's native currency (usually USD unless SET/etc)
        day_change_local_raw = stock_data.get('change')
        day_change_pct_raw = stock_data.get('changesPercentage')
        # prev_close_local_raw = stock_data.get('previousClose') # Not strictly needed for summary row

        # --- Price Source Logic ---
        # Determine price source and handle fallback
        price_source = "Unknown"
        current_price_local = np.nan
        day_change_local = np.nan
        day_change_pct = np.nan
        # prev_close_local = np.nan # Not strictly needed

        is_yahoo_price_valid = pd.notna(current_price_local_raw) and current_price_local_raw > 1e-9
        is_excluded = symbol in excluded_symbols

        if is_excluded:
            price_source = "Excluded - Fallback"
            current_price_local = 0.0 # Assume zero value if excluded
            is_yahoo_price_valid = False
        elif is_yahoo_price_valid:
            price_source = "Yahoo API/Cache"
            current_price_local = float(current_price_local_raw)
            if pd.notna(day_change_local_raw): day_change_local = float(day_change_local_raw)
            if pd.notna(day_change_pct_raw): day_change_pct = float(day_change_pct_raw)
            # if pd.notna(prev_close_local_raw): prev_close_local = float(prev_close_local_raw)
        else: # Price from Yahoo is invalid or missing
            price_source = "Yahoo Invalid - Fallback"
            current_price_local = 0.0 # Default to zero before trying fallback
            is_yahoo_price_valid = False # Ensure this is false

        # Fallback to last transaction price if Yahoo price invalid/missing (and not excluded)
        if not is_yahoo_price_valid and not is_excluded:
            try:
                # Find latest transaction with a valid price for this holding
                symbol_account_tx = transactions_df[
                    (transactions_df['Symbol'] == symbol) &
                    (transactions_df['Account'] == account) &
                    (transactions_df['Price/Share'].notna()) &
                    (transactions_df['Price/Share'] > 0)
                ] # Ensure Date column is datetime before using .loc
                if not symbol_account_tx.empty:
                    last_tx_row = symbol_account_tx.loc[symbol_account_tx['Date'].idxmax()] # Find row with latest date
                    last_tx_price = pd.to_numeric(last_tx_row['Price/Share'], errors='coerce')
                    if pd.notna(last_tx_price) and last_tx_price > 0:
                        current_price_local = float(last_tx_price)
                        price_source = "Last TX Fallback"
                    # else: keep current_price_local as 0.0 if last TX price invalid
            except Exception as e_fallback:
                # print(f"WARN: Fallback price lookup failed for {symbol}/{account}: {e_fallback}") # Optional
                price_source = "Fallback Zero (Error)"
                current_price_local = 0.0 # Ensure zero on error

            # Ensure final price is float
            try: current_price_local = float(current_price_local)
            except (ValueError, TypeError): current_price_local = 0.0
            # Reset day change if using fallback price
            day_change_local = np.nan
            day_change_pct = np.nan

        # --- Currency Conversion ---
        # Get rate for Local -> Display
        fx_rate = get_conversion_rate(local_currency, display_currency, current_fx_rates_vs_usd)
        if pd.isna(fx_rate): # Handle conversion failure
             print(f"ERROR: Failed to get FX rate {local_currency}->{display_currency} for {symbol}. Skipping summary row value calcs.")
             # Optionally add a row with NaNs or skip entirely
             # Let's add row with NaNs for now
             fx_rate = np.nan # Ensure it's NaN for calculations below

        # --- Calculate Display Currency Values ---
        market_value_local = current_qty * current_price_local
        account_market_values_local[account] += market_value_local # Aggregate local MV

        # Convert values to display currency
        market_value_display = market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        day_change_value_display = (current_qty * day_change_local * fx_rate) if pd.notna(day_change_local) and pd.notna(fx_rate) else np.nan
        current_price_display = current_price_local * fx_rate if pd.notna(current_price_local) and pd.notna(fx_rate) else np.nan
        cost_basis_display = np.nan
        avg_cost_price_display = np.nan
        unrealized_gain_display = np.nan
        unrealized_gain_pct = np.nan

        # --- Unrealized Gain/Loss Calculation ---
        is_long = current_qty > 1e-9
        is_short = current_qty < -1e-9

        if is_long:
            cost_basis_display = max(0, current_total_cost_local * fx_rate) if pd.notna(fx_rate) else np.nan # Cost basis cannot be negative for long
            if pd.notna(cost_basis_display):
                 avg_cost_price_display = (cost_basis_display / current_qty) if abs(current_qty) > 1e-9 else np.nan
                 if pd.notna(market_value_display):
                     unrealized_gain_display = market_value_display - cost_basis_display
                     if abs(cost_basis_display) > 1e-9:
                         unrealized_gain_pct = (unrealized_gain_display / cost_basis_display) * 100.0
                     # Handle zero cost basis case for percentage
                     elif abs(market_value_display) > 1e-9: unrealized_gain_pct = np.inf
                     else: unrealized_gain_pct = 0.0
        elif is_short:
            avg_cost_price_display = np.nan # Avg cost not typical for short
            cost_basis_display = 0.0 # Cost basis is zero for short sale itself
            short_proceeds_display = short_proceeds_local * fx_rate if pd.notna(fx_rate) else np.nan
            if pd.notna(market_value_display) and pd.notna(short_proceeds_display):
                # Unrealized Gain = Proceeds Received - Current Cost to Cover
                current_cost_to_cover_display = abs(market_value_display) # MV is negative, cost to cover is positive
                unrealized_gain_display = short_proceeds_display - current_cost_to_cover_display
                if abs(short_proceeds_display) > 1e-9:
                    unrealized_gain_pct = (unrealized_gain_display / short_proceeds_display) * 100.0
                # Handle zero proceeds case
                elif abs(current_cost_to_cover_display) > 1e-9: unrealized_gain_pct = -np.inf # Loss if cost > 0
                else: unrealized_gain_pct = 0.0

        # --- Other Display Values ---
        realized_gain_display = realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
        dividends_display = dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
        commissions_display = commissions_local * fx_rate if pd.notna(fx_rate) else np.nan

        # Total Gain
        unrealized_gain_component = unrealized_gain_display if pd.notna(unrealized_gain_display) else 0.0
        total_gain_display = (realized_gain_display + unrealized_gain_component + dividends_display - commissions_display) \
                             if all(pd.notna(v) for v in [realized_gain_display, dividends_display, commissions_display]) \
                             else np.nan

        # Total Return %
        total_cost_invested_display = total_cost_invested_local * fx_rate if pd.notna(fx_rate) else np.nan
        cumulative_investment_display = cumulative_investment_local * fx_rate if pd.notna(fx_rate) else np.nan
        total_return_pct = np.nan
        denominator_for_pct = cumulative_investment_display # Use cumulative investment for total return %
        if pd.notna(total_gain_display) and pd.notna(denominator_for_pct):
            if abs(denominator_for_pct) > 1e-9:
                total_return_pct = (total_gain_display / denominator_for_pct) * 100.0
            elif abs(total_gain_display) <= 1e-9 : # Gain is zero, cost is zero
                total_return_pct = 0.0
            # else: Gain is non-zero, cost is zero -> Inf or -Inf handled implicitly by float division or keep NaN

        # --- Calculate IRR ---
        # Use market value in LOCAL currency for IRR cash flow calculation
        market_value_local_for_irr = abs(market_value_local) if abs(current_qty) > 1e-9 else 0.0
        stock_irr = np.nan
        try:
            # Ensure transactions_df has the necessary columns and date format
            cf_dates, cf_values = get_cash_flows_for_symbol_account(
                symbol, account, transactions_df, market_value_local_for_irr, report_date
            )
            if cf_dates and cf_values:
                stock_irr = calculate_irr(cf_dates, cf_values)
        except Exception as e_irr:
            # print(f"WARN: IRR calculation failed for {symbol}/{account}: {e_irr}") # Optional warning
            stock_irr = np.nan

        irr_value_to_store = stock_irr * 100.0 if pd.notna(stock_irr) else np.nan

        # --- Append row data ---
        portfolio_summary_rows.append({
            'Account': account,
            'Symbol': symbol,
            'Quantity': current_qty,
            f'Avg Cost ({display_currency})': avg_cost_price_display,
            f'Price ({display_currency})': current_price_display,
            f'Cost Basis ({display_currency})': cost_basis_display,
            f'Market Value ({display_currency})': market_value_display,
            f'Day Change ({display_currency})': day_change_value_display, # Keep NaN if calc failed
            'Day Change %': day_change_pct, # Keep NaN if calc failed
            f'Unreal. Gain ({display_currency})': unrealized_gain_display,
            'Unreal. Gain %': unrealized_gain_pct,
            f'Realized Gain ({display_currency})': realized_gain_display,
            f'Dividends ({display_currency})': dividends_display,
            f'Commissions ({display_currency})': commissions_display,
            f'Total Gain ({display_currency})': total_gain_display,
            f'Total Cost Invested ({display_currency})': total_cost_invested_display, # Usually not shown
            'Total Return %': total_return_pct,
            f'Cumulative Investment ({display_currency})': cumulative_investment_display, # Needed for Total Return % calc
            'IRR (%)': irr_value_to_store,
            'Local Currency': local_currency, # Keep for reference
            'Price Source': price_source
        })
        # --- End of code to MOVE (Stock Holding Loop) ---

    # --- Loop 2: Process CASH Balances ---
    if cash_summary:
        for account, cash_data in cash_summary.items():
            # --- Start of code to MOVE (Cash Loop) ---
            symbol = CASH_SYMBOL_CSV
            # Get cash details (local currency)
            current_qty = cash_data.get('qty', 0.0)
            local_currency = cash_data.get('currency', default_currency)
            realized_gain_local = cash_data.get('realized', 0.0) # Usually 0 for cash
            dividends_local = cash_data.get('dividends', 0.0) # e.g. interest paid?
            commissions_local = cash_data.get('commissions', 0.0)

            # Store account currency mapping (might be redundant but safe)
            account_local_currency_map[account] = local_currency

            # --- Currency Conversion ---
            fx_rate = get_conversion_rate(local_currency, display_currency, current_fx_rates_vs_usd)
            if pd.isna(fx_rate):
                 print(f"ERROR: Failed to get FX rate {local_currency}->{display_currency} for {symbol} in {account}. Skipping summary row value calcs.")
                 fx_rate = np.nan # Ensure NaN for calculations

            # --- Calculate Display Currency Values for Cash ---
            current_price_local = 1.0 # Price of cash is 1 in its own currency
            market_value_local = current_qty * current_price_local
            account_market_values_local[account] += market_value_local # Aggregate local MV

            current_price_display = current_price_local * fx_rate if pd.notna(fx_rate) else np.nan
            market_value_display = market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
            # Cash specific values
            cost_basis_display = market_value_display # Cost basis of cash is its value
            avg_cost_price_display = current_price_display
            day_change_value_display = 0.0 # Assume no day change for cash
            day_change_pct = 0.0
            unrealized_gain_display = 0.0
            unrealized_gain_pct = 0.0
            # Convert other local values
            realized_gain_display = realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
            dividends_display = dividends_local * fx_rate if pd.notna(fx_rate) else np.nan # Interest
            commissions_display = commissions_local * fx_rate if pd.notna(fx_rate) else np.nan

            # Total Gain for cash (primarily interest - fees)
            total_gain_display = (dividends_display - commissions_display) if pd.notna(dividends_display) and pd.notna(commissions_display) else np.nan

            # Total Return % for Cash (based on interest vs avg balance? Complex, set NaN for now)
            # Cumulative Investment for cash is just its current value if positive, or relates to deposits/withdrawals.
            # For simplicity in summary row, let's mirror market value.
            cumulative_investment_display = market_value_display if pd.notna(market_value_display) else np.nan
            total_return_pct_cash = np.nan # Hard to define simply
   
            # --- Append cash row data ---
            portfolio_summary_rows.append({
                'Account': account,
                'Symbol': symbol,
                'Quantity': current_qty,
                f'Avg Cost ({display_currency})': avg_cost_price_display,
                f'Price ({display_currency})': current_price_display,
                f'Cost Basis ({display_currency})': cost_basis_display,
                f'Market Value ({display_currency})': market_value_display,
                f'Day Change ({display_currency})': day_change_value_display,
                'Day Change %': day_change_pct,
                f'Unreal. Gain ({display_currency})': unrealized_gain_display,
                'Unreal. Gain %': unrealized_gain_pct,
                f'Realized Gain ({display_currency})': realized_gain_display, # Usually 0
                f'Dividends ({display_currency})': dividends_display, # Interest
                f'Commissions ({display_currency})': commissions_display,
                f'Total Gain ({display_currency})': total_gain_display,
                f'Total Cost Invested ({display_currency})': cost_basis_display, # Use cost basis for cash invested
                'Total Return %': total_return_pct_cash, # Set to NaN
                f'Cumulative Investment ({display_currency})': cumulative_investment_display, # For consistency
                'IRR (%)': np.nan, # IRR not applicable to cash balance itself
                'Local Currency': local_currency,
                'Price Source': 'N/A (Cash)'
            })
            # --- End of code to MOVE (Cash Loop) ---

    # Convert defaultdicts to dicts before returning
    return portfolio_summary_rows, dict(account_market_values_local), dict(account_local_currency_map)

# Revised safe_sum - Attempt 3 (Direct Sum on Subset)
def safe_sum(df, col):
    """Safely sums a DataFrame column, handling NaNs."""
    if col not in df.columns:
        return 0.0 # Column doesn't exist

    try:
        # Select the column, convert errors to NaN, fill NaN with 0, then sum
        # This ensures we are working with a Series before sum
        data_series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        total = data_series.sum()
        # Ensure return is a standard float, handle potential lingering NaNs from sum itself
        return float(total) if pd.notna(total) else 0.0
    except Exception as e:
        # print(f"Error in safe_sum for column {col}: {e}") # Optional debug
        return 0.0 # Return 0 on any unexpected error during sum
               
def _calculate_aggregate_metrics(
    full_summary_df: pd.DataFrame,
    display_currency: str,
    transactions_df: pd.DataFrame, # Filtered tx df for MWR
    report_date: date,
    # Pass the necessary data for MWR's FX lookups - requires historical FX
    # This is tricky - calculate_portfolio_summary doesn't fetch historical FX.
    # Let's postpone MWR calculation here and focus on summing totals first.
    # We can add MWR back later or decide it belongs elsewhere.
    # ---- OR ----
    # If MWR *must* be calculated here using CURRENT rates (less accurate):
    current_fx_rates_vs_usd: Dict[str, float], # If using current rates for MWR
    account_local_currency_map: Dict[str, str] # Needed for MWR flow conversion if using current rates

) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Calculates account-level and overall portfolio summary metrics
    (Totals, Day Change %) from the full summary DataFrame.
    Note: MWR calculation using current rates might be inaccurate. Consider
          calculating MWR alongside historical TWR if historical rates are needed.

    Args:
        full_summary_df: DataFrame containing detailed rows for all holdings/cash.
        display_currency: The currency used in the DataFrame's monetary columns.
        transactions_df: The filtered transaction DataFrame (potentially needed for MWR).
        report_date: The date for which the summary applies.
        current_fx_rates_vs_usd: Dictionary of CURR per USD rates (for MWR if using current).
        account_local_currency_map: Map of account to local currency (for MWR if using current).


    Returns:
        A tuple containing:
        - overall_summary_metrics: Dict of aggregated metrics for the whole scope.
        - account_level_metrics: Dict keyed by account with account-level aggregates.
    """
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        'mwr': np.nan, 'total_return_pct': np.nan, 'total_market_value_display': 0.0,
        'total_realized_gain_display': 0.0, 'total_unrealized_gain_display': 0.0,
        'total_dividends_display': 0.0, 'total_commissions_display': 0.0,
        'total_gain_display': 0.0, 'total_cash_display': 0.0,
        'total_cost_invested_display': 0.0, 'total_day_change_display': 0.0,
        'total_day_change_percent': np.nan
    })
    overall_summary_metrics = {} # Initialize

    print("Calculating Account-Level & Overall Metrics...") # Update print

    # --- Start of code to MOVE ---
    unique_accounts_in_summary = full_summary_df['Account'].unique()
    for account in unique_accounts_in_summary:
        account_full_df = full_summary_df[full_summary_df['Account'] == account]
        metrics_entry = account_level_metrics[account] # Get the defaultdict entry

        # --- Calculate Account MWR (Using CURRENT Rates - Check if appropriate) ---
        # Note: Passing current_fx_rates_vs_usd to get_cash_flows_for_mwr assumes we want
        # MWR based on today's rates, which might differ from historical calculations.
        # Consider if this MWR calculation is needed here or should use historical data.
        account_tx = transactions_df[transactions_df['Account'] == account]
        # Account MV already calculated and stored in account_full_df
        account_mv_display = safe_sum(account_full_df, f'Market Value ({display_currency})')
        metrics_entry['total_market_value_display'] = account_mv_display
        mwr = np.nan
        try:
            # This needs the standard {A/B: rate} dict, not CURR/USD dict
            # We need to recalculate the full rates dict if we only have CURR/USD
            # OR modify get_cash_flows_for_mwr to accept CURR/USD dict.
            # --> Let's skip Account MWR here for now to simplify refactoring <--
            # print(f"Skipping Account MWR calculation for {account} in this refactoring step.")
            # Pass the account_mv_display (already converted)
            # cf_dates_mwr, cf_values_mwr = get_cash_flows_for_mwr(
            #     account_tx,
            #     account_mv_display, # Final value in display currency
            #     report_date,
            #     display_currency, # Target currency is display currency
            #     current_fx_rates_standard, # <<< Needs the standard A/B dict!
            #     display_currency
            # )
            # if cf_dates_mwr and cf_values_mwr:
            #     mwr = calculate_irr(cf_dates_mwr, cf_values_mwr)
            metrics_entry['mwr'] = np.nan # Set to NaN for now
        except Exception as e_mwr:
            # print(f"WARN: Account MWR calc failed for {account}: {e_mwr}") # Optional
            metrics_entry['mwr'] = np.nan

        # --- Aggregate Account Totals ---
        metrics_entry['total_realized_gain_display'] = safe_sum(account_full_df, f'Realized Gain ({display_currency})')
        metrics_entry['total_unrealized_gain_display'] = safe_sum(account_full_df, f'Unreal. Gain ({display_currency})')
        metrics_entry['total_dividends_display'] = safe_sum(account_full_df, f'Dividends ({display_currency})')
        metrics_entry['total_commissions_display'] = safe_sum(account_full_df, f'Commissions ({display_currency})')
        metrics_entry['total_gain_display'] = safe_sum(account_full_df, f'Total Gain ({display_currency})')
        # Ensure CASH_SYMBOL_CSV is used here
        cash_symbol = CASH_SYMBOL_CSV
        metrics_entry['total_cash_display'] = safe_sum(account_full_df[account_full_df['Symbol'] == cash_symbol], f'Market Value ({display_currency})')
        metrics_entry['total_cost_invested_display'] = safe_sum(account_full_df, f'Total Cost Invested ({display_currency})') # Use safe_sum
        acc_cumulative_investment_display = safe_sum(account_full_df, f'Cumulative Investment ({display_currency})')

        # Account Total Return %
        acc_total_gain = metrics_entry['total_gain_display']
        acc_denominator = acc_cumulative_investment_display
        acc_total_return_pct = np.nan
        if pd.notna(acc_total_gain) and pd.notna(acc_denominator):
             if abs(acc_denominator) > 1e-9:
                 acc_total_return_pct = (acc_total_gain / acc_denominator) * 100.0
             elif abs(acc_total_gain) <= 1e-9:
                 acc_total_return_pct = 0.0
             # else leave NaN if gain != 0 and denominator == 0
        metrics_entry['total_return_pct'] = acc_total_return_pct

        # Account Day Change %
        acc_total_day_change_display = safe_sum(account_full_df, f'Day Change ({display_currency})')
        metrics_entry['total_day_change_display'] = acc_total_day_change_display
        acc_current_mv_display = metrics_entry['total_market_value_display'] # Already calculated
        acc_prev_close_mv_display = np.nan
        if pd.notna(acc_current_mv_display) and pd.notna(acc_total_day_change_display):
             acc_prev_close_mv_display = acc_current_mv_display - acc_total_day_change_display

        metrics_entry['total_day_change_percent'] = np.nan
        if pd.notna(acc_total_day_change_display) and pd.notna(acc_prev_close_mv_display):
            if abs(acc_prev_close_mv_display) > 1e-9:
                 try: metrics_entry['total_day_change_percent'] = (acc_total_day_change_display / acc_prev_close_mv_display) * 100.0
                 except ZeroDivisionError: pass
            elif abs(acc_total_day_change_display) > 1e-9: # Change exists but starting value was zero
                metrics_entry['total_day_change_percent'] = np.inf if acc_total_day_change_display > 0 else -np.inf
            elif abs(acc_total_day_change_display) < 1e-9 : # Zero change and zero starting value
                metrics_entry['total_day_change_percent'] = 0.0

    # --- Overall metrics ---
    # Use safe_sum on the full DataFrame columns
    overall_market_value_display = safe_sum(full_summary_df, f'Market Value ({display_currency})')
    # Cost basis only for currently held positions (non-zero qty or cash)
    held_mask = (full_summary_df['Quantity'].abs() > 1e-9) | (full_summary_df['Symbol'] == CASH_SYMBOL_CSV)
    overall_cost_basis_display = safe_sum(full_summary_df.loc[held_mask], f'Cost Basis ({display_currency})')
    overall_unrealized_gain_display = safe_sum(full_summary_df, f'Unreal. Gain ({display_currency})')
    overall_realized_gain_display_agg = safe_sum(full_summary_df, f'Realized Gain ({display_currency})')
    overall_dividends_display_agg = safe_sum(full_summary_df, f'Dividends ({display_currency})')
    overall_commissions_display_agg = safe_sum(full_summary_df, f'Commissions ({display_currency})')
    overall_total_gain_display = safe_sum(full_summary_df, f'Total Gain ({display_currency})')
    overall_total_cost_invested_display = safe_sum(full_summary_df, f'Total Cost Invested ({display_currency})')
    overall_cumulative_investment_display = safe_sum(full_summary_df, f'Cumulative Investment ({display_currency})')

    # Overall MWR - Skipping for now in this refactoring step (same FX rate issue as account MWR)
    overall_portfolio_mwr = np.nan
    # try:
    #     cf_dates_overall, cf_values_overall = get_cash_flows_for_mwr(
    #          transactions_df, # Use the filtered transaction df for the scope
    #          overall_market_value_display, report_date, display_currency,
    #          current_fx_rates_standard, # <<< Needs the standard A/B dict!
    #          display_currency
    #     )
    #     if cf_dates_overall and cf_values_overall:
    #          overall_portfolio_mwr = calculate_irr(cf_dates_overall, cf_values_overall)
    # except Exception as e_mwr_overall: overall_portfolio_mwr = np.nan

    # Overall Day Change %
    overall_day_change_display = safe_sum(full_summary_df, f'Day Change ({display_currency})')
    overall_prev_close_mv_display = np.nan
    if pd.notna(overall_market_value_display) and pd.notna(overall_day_change_display):
         overall_prev_close_mv_display = overall_market_value_display - overall_day_change_display

    overall_day_change_percent = np.nan
    if pd.notna(overall_day_change_display) and pd.notna(overall_prev_close_mv_display):
        if abs(overall_prev_close_mv_display) > 1e-9:
             try: overall_day_change_percent = (overall_day_change_display / overall_prev_close_mv_display) * 100.0
             except ZeroDivisionError: pass
        elif abs(overall_day_change_display) > 1e-9:
            overall_day_change_percent = np.inf if overall_day_change_display > 0 else -np.inf
        elif abs(overall_day_change_display) < 1e-9 :
            overall_day_change_percent = 0.0

    # Overall Total Return %
    overall_total_return_pct = np.nan
    if pd.notna(overall_total_gain_display) and pd.notna(overall_cumulative_investment_display):
        if abs(overall_cumulative_investment_display) > 1e-9:
            overall_total_return_pct = (overall_total_gain_display / overall_cumulative_investment_display) * 100.0
        elif abs(overall_total_gain_display) <= 1e-9:
            overall_total_return_pct = 0.0

    overall_summary_metrics = {
        "market_value": overall_market_value_display,
        "cost_basis_held": overall_cost_basis_display,
        "unrealized_gain": overall_unrealized_gain_display,
        "realized_gain": overall_realized_gain_display_agg,
        "dividends": overall_dividends_display_agg,
        "commissions": overall_commissions_display_agg,
        "total_gain": overall_total_gain_display,
        "total_cost_invested": overall_total_cost_invested_display, # Maybe less useful?
        "portfolio_mwr": np.nan, # Set MWR to NaN for now
        "day_change_display": overall_day_change_display,
        "day_change_percent": overall_day_change_percent,
        "report_date": report_date.strftime('%Y-%m-%d'),
        "display_currency": display_currency,
        "cumulative_investment": overall_cumulative_investment_display,
        "total_return_pct": overall_total_return_pct
    }
    # --- End of code to MOVE ---

    # Convert defaultdict back to dict for return
    return overall_summary_metrics, dict(account_level_metrics)

# --- Main Calculation Function (Current Portfolio Summary) ---
# Assume helper functions (_load_and_clean_transactions, _process_transactions_to_holdings,
# _calculate_cash_balances, _build_summary_rows, _calculate_aggregate_metrics, safe_sum,
# get_cached_or_fetch_yfinance_data, get_conversion_rate, get_cash_flows_for_mwr,
# calculate_irr, calculate_npv) are defined above this function.

# Assume constants like CASH_SYMBOL_CSV_PERFORM, SHORTABLE_SYMBOLS, YFINANCE_EXCLUDED_SYMBOLS are defined globally.

def calculate_portfolio_summary(
    transactions_csv_file: str,
    fmp_api_key: Optional[str] = None, # Currently unused but kept for signature
    display_currency: str = 'USD',
    show_closed_positions: bool = False,
    account_currency_map: Dict = {'SET': 'THB'}, # Default for signature
    default_currency: str = 'USD',             # Default for signature
    cache_file_path: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    include_accounts: Optional[List[str]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict[str, Dict[str, float]]], str]:
    """
    Calculates portfolio summary using Yahoo Finance for current data.
    Filters calculations based on `include_accounts`.
    Uses `account_currency_map` and `default_currency`. Calls helper functions
    for processing steps.

    Args:
        transactions_csv_file (str): Path to the transactions CSV file.
        fmp_api_key (Optional[str]): FMP API key (currently unused).
        display_currency (str): The currency for displaying results.
        show_closed_positions (bool): Whether to include positions with zero quantity.
        account_currency_map (Dict): Map of account names to their local currency.
        default_currency (str): Default currency if account not in map.
        cache_file_path (str): Path for the Yahoo Finance current data cache.
        include_accounts (Optional[List[str]]): List of account names to include. If None/empty, all accounts included.

    Returns:
        Tuple:
            - Dict: Overall summary metrics for the included accounts (or None on failure).
            - DataFrame: Detailed holdings for the included accounts (or None).
            - DataFrame: Ignored transactions (or None).
            - Dict: Account-level metrics for included accounts (or None).
            - str: Status message string.
    """
    print(f"\n--- Starting Portfolio Calculation (Yahoo Finance - Current Summary) ---")
    filter_desc = "All Accounts"
    if include_accounts: filter_desc = f"Accounts: {', '.join(sorted(include_accounts))}"
    print(f"Parameters: CSV='{os.path.basename(transactions_csv_file)}', Currency='{display_currency}', ShowClosed={show_closed_positions}, Scope='{filter_desc}'")
    print(f"Currency Settings: Default='{default_currency}', Account Map='{account_currency_map}'")

    original_transactions_df: Optional[pd.DataFrame] = None
    all_transactions_df: Optional[pd.DataFrame] = None
    ignored_row_indices = set()
    ignored_reasons: Dict[int, str] = {}
    status_messages = []
    report_date = datetime.now().date() # Use current date for summary report

    # --- 1. Load & Clean ALL Transactions ---
    all_transactions_df, original_transactions_df, ignored_indices_load, ignored_reasons_load = _load_and_clean_transactions(
        transactions_csv_file,
        account_currency_map,
        default_currency
    )
    ignored_row_indices.update(ignored_indices_load)
    ignored_reasons.update(ignored_reasons_load)

    if all_transactions_df is None:
        msg = f"Error: File not found or failed to load/clean: {transactions_csv_file}"
        # Attempt to return original if load failed but read occurred
        ignored_df_final = original_transactions_df.loc[sorted(list(ignored_row_indices))].copy() if ignored_row_indices and original_transactions_df is not None else pd.DataFrame()
        return None, None, ignored_df_final, None, msg
    if ignored_reasons: status_messages.append(f"Info: {len(ignored_reasons)} transactions ignored during load/clean.")

    # Get available accounts from the full dataset
    all_available_accounts_list = []
    if 'Account' in all_transactions_df.columns:
        all_available_accounts_list = sorted(all_transactions_df['Account'].unique().tolist())

    # --- 2. Filter Transactions based on include_accounts ---
    transactions_df = pd.DataFrame() # This will hold the filtered transactions for processing
    available_accounts_set = set(all_available_accounts_list)
    if not include_accounts: # If None or empty list, use all
        print("Info: No specific accounts provided, using all available accounts.")
        transactions_df = all_transactions_df.copy()
        included_accounts_list = sorted(list(available_accounts_set))
    else:
        valid_include_accounts = [acc for acc in include_accounts if acc in available_accounts_set]
        if not valid_include_accounts:
            msg = "Warning: None of the specified accounts to include were found. No data processed."
            print(msg); status_messages.append(msg)
            ignored_df_final = original_transactions_df.loc[sorted(list(ignored_row_indices))].copy() if ignored_row_indices and original_transactions_df is not None else pd.DataFrame()
            return {}, pd.DataFrame(), ignored_df_final, {}, msg # Return empty but valid structures
        print(f"Info: Filtering transactions FOR accounts: {', '.join(sorted(valid_include_accounts))}")
        transactions_df = all_transactions_df[all_transactions_df['Account'].isin(valid_include_accounts)].copy()
        included_accounts_list = sorted(valid_include_accounts) # Use the validated list

    if transactions_df.empty:
        msg = f"Warning: No transactions remain after filtering for accounts: {', '.join(sorted(include_accounts if include_accounts else []))}"
        print(msg); status_messages.append(msg)
        ignored_df_final = original_transactions_df.loc[sorted(list(ignored_row_indices))].copy() if ignored_row_indices and original_transactions_df is not None else pd.DataFrame()
        # Add available accounts to empty summary for GUI consistency
        empty_summary = {'_available_accounts': all_available_accounts_list}
        return empty_summary, pd.DataFrame(), ignored_df_final, {}, msg

    # --- 3. Process Stock/ETF Transactions ---
    holdings, overall_realized_gains_local, overall_dividends_local, overall_commissions_local, \
    ignored_indices_proc, ignored_reasons_proc = _process_transactions_to_holdings(
        transactions_df=transactions_df,
        default_currency=default_currency,
        shortable_symbols=SHORTABLE_SYMBOLS
    )
    ignored_row_indices.update(ignored_indices_proc)
    ignored_reasons.update(ignored_reasons_proc)

    # --- 4. Calculate $CASH Balances ---
    cash_summary = _calculate_cash_balances(
        transactions_df=transactions_df,
        default_currency=default_currency
    )

    # --- 5. Fetch Current Market Data (Prices, Change, FX vs USD) ---
    print("Determining required data and fetching from Yahoo Finance...")
    all_stock_symbols_internal = list(set(key[0] for key in holdings.keys())) # No need for cash symbol here
    # Determine required currencies from holdings, cash summary, display, and default
    required_currencies: Set[str] = set([display_currency, default_currency])
    for data in holdings.values(): required_currencies.add(data.get('local_currency', default_currency))
    for data in cash_summary.values(): required_currencies.add(data.get('currency', default_currency))
    required_currencies.discard('N/A') # Remove potential N/A if it sneaks in
    required_currencies.discard(None)

    current_stock_data_internal, current_fx_rates_vs_usd = get_cached_or_fetch_yfinance_data(
        internal_stock_symbols=all_stock_symbols_internal,
        required_currencies=required_currencies,
        cache_file=cache_file_path
    )

    if current_stock_data_internal is None or current_fx_rates_vs_usd is None:
        msg = "Error: Price/FX fetch failed via Yahoo Finance. Cannot calculate current values."
        print(f"FATAL: {msg}")
        status_messages.append(msg)
        ignored_df_final = original_transactions_df.loc[sorted(list(ignored_row_indices))].copy() if ignored_row_indices and original_transactions_df is not None else pd.DataFrame()
        # Return None for metrics, DataFrames; include available accounts in empty summary
        empty_summary = {'_available_accounts': all_available_accounts_list}
        return empty_summary, pd.DataFrame(), ignored_df_final, {}, msg

    # --- 6. Build Detailed Summary Rows (in display currency) ---
    portfolio_summary_rows, account_market_values_local, account_local_currency_map = _build_summary_rows(
        holdings=holdings,
        cash_summary=cash_summary,
        current_stock_data=current_stock_data_internal,
        current_fx_rates_vs_usd=current_fx_rates_vs_usd, # Pass CURR per USD rates
        display_currency=display_currency,
        default_currency=default_currency,
        transactions_df=transactions_df, # Pass filtered transactions for IRR
        report_date=report_date,
        shortable_symbols=SHORTABLE_SYMBOLS,
        excluded_symbols=YFINANCE_EXCLUDED_SYMBOLS
    )

    # --- Initialize return values and metrics dicts ---
    summary_df = pd.DataFrame()
    summary_df_for_return = pd.DataFrame() # Initialize here
    overall_summary_metrics = {}
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: { # Initialize account metrics
         'mwr': np.nan, 'total_return_pct': np.nan, 'total_market_value_display': 0.0,
         'total_realized_gain_display': 0.0, 'total_unrealized_gain_display': 0.0,
         'total_dividends_display': 0.0, 'total_commissions_display': 0.0,
         'total_gain_display': 0.0, 'total_cash_display': 0.0,
         'total_cost_invested_display': 0.0, 'total_day_change_display': 0.0,
         'total_day_change_percent': np.nan
    })

    # --- 7. Create DataFrame & Calculate Aggregates ---
    if not portfolio_summary_rows:
        print("Warning: Portfolio summary list is empty after processing holdings and cash.")
        overall_summary_metrics = { # Populate with zeros/NaNs
             "market_value": 0.0, "cost_basis_held": 0.0, "unrealized_gain": 0.0,
             "realized_gain": 0.0, "dividends": 0.0, "commissions": 0.0,
             "total_gain": 0.0, "total_cost_invested": 0.0, "portfolio_mwr": np.nan,
             "day_change_display": 0.0, "day_change_percent": np.nan,
             "report_date": report_date.strftime('%Y-%m-%d'),
             "display_currency": display_currency, "cumulative_investment": 0.0,
             "total_return_pct": np.nan
        }
        status_messages.append("Warning: No holdings data generated for selected accounts.")
        # summary_df remains empty, account_level_metrics remains empty defaultdict
    else:
        full_summary_df = pd.DataFrame(portfolio_summary_rows)

        # Convert data types
        money_cols_display = [c for c in full_summary_df.columns if f'({display_currency})' in c]
        percent_cols = ['Unreal. Gain %', 'Total Return %', 'IRR (%)', 'Day Change %']
        numeric_cols_to_convert = ['Quantity'] + money_cols_display + percent_cols
        if f'Cumulative Investment ({display_currency})' not in numeric_cols_to_convert:
             numeric_cols_to_convert.append(f'Cumulative Investment ({display_currency})')
        for col in numeric_cols_to_convert:
            if col in full_summary_df.columns:
                full_summary_df[col] = pd.to_numeric(full_summary_df[col], errors='coerce')

        # Sort
        try: # Sorting might fail if essential columns are missing/all NaN
             full_summary_df.sort_values(by=['Account', f'Market Value ({display_currency})'], ascending=[True, False], na_position='last', inplace=True)
        except KeyError:
             print("Warning: Could not sort summary DataFrame (required columns might be missing).")


        # Calculate Aggregates using the helper
        cash_symbol = CASH_SYMBOL_CSV # Ensure defined
        overall_summary_metrics, account_level_metrics = _calculate_aggregate_metrics(
            full_summary_df=full_summary_df,
            display_currency=display_currency,
            transactions_df=transactions_df,
            report_date=report_date,
            current_fx_rates_vs_usd=current_fx_rates_vs_usd, # Pass if MWR uses current rates
            account_local_currency_map=account_local_currency_map # Pass if MWR uses current rates
        )

        # Prepare the DataFrame to be returned, potentially filtering closed positions
        summary_df_for_return = full_summary_df # Start with the full one
        if not show_closed_positions:
             held_mask = (full_summary_df['Quantity'].abs() > 1e-9) | (full_summary_df['Symbol'] == cash_symbol)
             summary_df_for_return = full_summary_df[held_mask].copy()

    # --- Final Assignment of DataFrame to Return ---
    summary_df = summary_df_for_return

    # --- Add Metadata to Overall Summary ---
    # Ensure overall_summary_metrics exists before adding keys
    if overall_summary_metrics is None: overall_summary_metrics = {}
    overall_summary_metrics['_available_accounts'] = all_available_accounts_list
    # Add exchange rate if needed
    if display_currency != default_currency and current_fx_rates_vs_usd:
         base_to_display_rate = get_conversion_rate(default_currency, display_currency, current_fx_rates_vs_usd)
         # Use isfinite to check for NaN or +/- inf explicitly
         if base_to_display_rate != 1.0 and np.isfinite(base_to_display_rate):
              overall_summary_metrics['exchange_rate_to_display'] = base_to_display_rate


    # --- 8. Prepare Ignored Transactions DataFrame ---
    ignored_df_final = pd.DataFrame()
    if ignored_row_indices and original_transactions_df is not None:
        # Filter original_df using valid indices present in its index
        valid_indices = sorted([idx for idx in ignored_row_indices if idx in original_transactions_df.index])
        if valid_indices:
             ignored_df_final = original_transactions_df.loc[valid_indices].copy()
             try:
                 # Map reasons using the combined ignored_reasons dictionary
                 ignored_df_final['Reason Ignored'] = ignored_df_final.index.map(ignored_reasons).fillna("Unknown Reason")
             except Exception as e_reason:
                 print(f"Warning: Could not add 'Reason Ignored' column: {e_reason}")


    # --- 9. Determine Final Status ---
    final_status = f"Success ({filter_desc})"
    price_source_warnings = False
    if not full_summary_df.empty and 'Price Source' in full_summary_df.columns:
         non_cash_holdings = full_summary_df[full_summary_df['Symbol'] != CASH_SYMBOL_CSV]
         if not non_cash_holdings.empty:
             price_source_warnings = non_cash_holdings['Price Source'].str.contains("Fallback|Excluded|Invalid", na=False).any()

    # Combine reasons from different stages
    all_reasons = list(ignored_reasons.values()) + status_messages
    warnings = any("WARN" in r.upper() for r in all_reasons) or price_source_warnings or ignored_row_indices
    errors = any("ERROR" in r.upper() for r in all_reasons) or any("FAIL" in r.upper() for r in all_reasons)
    critical_warnings = any("CRITICAL" in r.upper() for r in all_reasons)

    if critical_warnings: final_status = f"Success with Critical Warnings ({filter_desc})"
    elif errors: final_status = f"Success with Errors ({filter_desc})"
    elif warnings: final_status = f"Success with Warnings ({filter_desc})"

    print(f"--- Portfolio Calculation Finished ({filter_desc}) ---")
    # Convert defaultdict back to dict for account_metrics return
    return overall_summary_metrics, summary_df, ignored_df_final, dict(account_level_metrics), final_status

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

def _prepare_historical_inputs(
    transactions_csv_file: str,
    account_currency_map: Dict,
    default_currency: str,
    include_accounts: Optional[List[str]],
    exclude_accounts: Optional[List[str]],
    start_date: date,
    end_date: date,
    benchmark_symbols_yf: List[str], # Cleaned list
    display_currency: str,
    current_hist_version: str = "v10", # Pass version for cache keys
    raw_cache_prefix: str = HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, # Pass prefixes
    daily_cache_prefix: str = DAILY_RESULTS_CACHE_PATH_PREFIX
) -> Tuple[
        Optional[pd.DataFrame], # transactions_df_effective
        Optional[pd.DataFrame], # original_transactions_df
        Set[int], Dict[int, str], # ignored_indices, ignored_reasons
        List[str], # all_available_accounts_list
        List[str], # included_accounts_list_sorted
        List[str], # excluded_accounts_list_sorted
        List[str], # symbols_for_stocks_and_benchmarks_yf
        List[str], # fx_pairs_for_api_yf
        Dict[str, str], # internal_to_yf_map
        Dict[str, str], # yf_to_internal_map_hist
        Dict[str, List[Dict]], # splits_by_internal_symbol
        str, # raw_data_cache_file
        str, # raw_data_cache_key
        Optional[str], # daily_results_cache_file
        Optional[str], # daily_results_cache_key
        str # filter_desc
    ]:
    """
    Loads, cleans, filters transactions, determines required symbols/currencies/FX pairs,
    extracts splits, and generates cache keys/filenames for historical analysis.
    """
    print("Preparing inputs for historical calculation...")

    # Initialize return values for failure cases
    empty_tuple_return = (None, None, set(), {}, [], [], [], [], [], {}, {}, {}, "", "", None, None, "")

    # --- 1. Load & Clean ALL Transactions ---
    all_transactions_df, original_transactions_df, ignored_indices, ignored_reasons = _load_and_clean_transactions(
        transactions_csv_file, account_currency_map, default_currency
    )
    if all_transactions_df is None:
        print("ERROR in _prepare_historical_inputs: Failed to load/clean transactions.")
        return empty_tuple_return # Return empties/None on failure

    # --- Get available accounts ---
    all_available_accounts_list = []
    if 'Account' in all_transactions_df.columns:
        all_available_accounts_list = sorted(all_transactions_df['Account'].unique().tolist())

    # --- 1b. Filter Transactions ---
    transactions_df_effective = pd.DataFrame()
    available_accounts_set = set(all_available_accounts_list)
    included_accounts_list_sorted = []
    excluded_accounts_list_sorted = []
    filter_desc = "All Accounts" # Default description

    # Apply inclusion filter
    if not include_accounts:
        transactions_df_included = all_transactions_df.copy()
        included_accounts_list_sorted = sorted(list(available_accounts_set))
    else:
        valid_include_accounts = [acc for acc in include_accounts if acc in available_accounts_set]
        if not valid_include_accounts:
            print("WARN in _prepare_historical_inputs: No valid accounts to include.")
            return empty_tuple_return # Or maybe return partial data? For now, return empty.
        transactions_df_included = all_transactions_df[all_transactions_df['Account'].isin(valid_include_accounts)].copy()
        included_accounts_list_sorted = sorted(valid_include_accounts)
        filter_desc = f"Included: {', '.join(included_accounts_list_sorted)}"

    # Apply exclusion filter
    if not exclude_accounts or not isinstance(exclude_accounts, list):
        transactions_df_effective = transactions_df_included.copy()
    else:
        valid_exclude_accounts = [acc for acc in exclude_accounts if acc in available_accounts_set]
        if valid_exclude_accounts:
            print(f"Hist Prep: Excluding accounts: {', '.join(sorted(valid_exclude_accounts))}")
            transactions_df_effective = transactions_df_included[~transactions_df_included['Account'].isin(valid_exclude_accounts)].copy()
            excluded_accounts_list_sorted = sorted(valid_exclude_accounts)
            # Update description
            if include_accounts: filter_desc += f" (Excluding: {', '.join(excluded_accounts_list_sorted)})"
            else: filter_desc = f"All Accounts (Excluding: {', '.join(excluded_accounts_list_sorted)})"
        else: # No valid exclusions
            transactions_df_effective = transactions_df_included.copy()

    if transactions_df_effective.empty:
        print("WARN in _prepare_historical_inputs: No transactions remain after filtering.")
        # Return effective df as empty, but provide other info if possible
        return (transactions_df_effective, original_transactions_df, ignored_indices, ignored_reasons,
                all_available_accounts_list, included_accounts_list_sorted, excluded_accounts_list_sorted,
                [], [], {}, {}, {}, "", "", None, None, filter_desc)

    # --- Extract Split Information (use original full df for splits) ---
    split_transactions = all_transactions_df[all_transactions_df['Type'].str.lower().isin(['split', 'stock split']) & all_transactions_df['Split Ratio'].notna() & (all_transactions_df['Split Ratio'] > 0)].sort_values(by='Date', ascending=True)
    splits_by_internal_symbol = {
        symbol: group[['Date', 'Split Ratio']].apply(lambda r: {'Date': r['Date'].date(), 'Split Ratio': float(r['Split Ratio'])}, axis=1).tolist()
        for symbol, group in split_transactions.groupby('Symbol')
    }

    # --- Determine Required Symbols & FX Pairs (use EFFECTIVE df) ---
    all_symbols_internal = list(set(transactions_df_effective['Symbol'].unique()))
    symbols_to_fetch_yf_portfolio = []; internal_to_yf_map = {}; yf_to_internal_map_hist = {}
    for internal_sym in all_symbols_internal:
        if internal_sym == CASH_SYMBOL_CSV: continue
        yf_sym = map_to_yf_symbol(internal_sym); # Assumes map_to_yf_symbol helper exists
        if yf_sym:
            symbols_to_fetch_yf_portfolio.append(yf_sym);
            internal_to_yf_map[internal_sym] = yf_sym;
            yf_to_internal_map_hist[yf_sym] = internal_sym
    symbols_to_fetch_yf_portfolio = sorted(list(set(symbols_to_fetch_yf_portfolio)))
    symbols_for_stocks_and_benchmarks_yf = sorted(list(set(symbols_to_fetch_yf_portfolio + benchmark_symbols_yf))) # Use cleaned benchmark list passed in

    all_currencies_in_tx = set(transactions_df_effective['Local Currency'].unique());
    all_currencies_needed = all_currencies_in_tx.union({display_currency, default_currency}) # Add display and default
    all_currencies_needed.discard(None) # Remove None if present
    all_currencies_needed.discard('N/A') # Remove N/A

    # Generate necessary YF FX tickers (CURRENCY=X format)
    fx_pairs_for_api_yf = set()
    for curr in all_currencies_needed:
         if curr != 'USD':
             fx_pairs_for_api_yf.add(f"{curr}=X")
    fx_pairs_for_api_yf = sorted(list(fx_pairs_for_api_yf))

    # --- Generate Cache Keys & Filenames ---
    raw_data_cache_file = f"{raw_cache_prefix}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    raw_data_cache_key = f"ADJUSTED_v7::{start_date.isoformat()}::{end_date.isoformat()}::{'_'.join(sorted(symbols_for_stocks_and_benchmarks_yf))}::{'_'.join(fx_pairs_for_api_yf)}" # Use correct FX list

    daily_results_cache_file = None
    daily_results_cache_key = None
    try:
        tx_file_hash = _get_file_hash(transactions_csv_file) # Assumes helper exists
        acc_map_str = json.dumps(account_currency_map, sort_keys=True)
        included_accounts_str = json.dumps(included_accounts_list_sorted)
        excluded_accounts_str = json.dumps(excluded_accounts_list_sorted)
        daily_results_cache_key = f"DAILY_RES_{current_hist_version}::{start_date.isoformat()}::{end_date.isoformat()}::{tx_file_hash}::{'_'.join(sorted(benchmark_symbols_yf))}::{display_currency}::{acc_map_str}::{default_currency}::{included_accounts_str}::{excluded_accounts_str}"
        cache_key_hash = hashlib.sha256(daily_results_cache_key.encode()).hexdigest()[:16] # Use first 16 chars of hash
        daily_results_cache_file = f"{daily_cache_prefix}_{cache_key_hash}.json"
    except Exception as e_key:
        print(f"Hist Prep WARN: Could not generate daily results cache key/filename: {e_key}.")
        # Keep them as None

    return (
        transactions_df_effective, original_transactions_df, ignored_indices, ignored_reasons,
        all_available_accounts_list, included_accounts_list_sorted, excluded_accounts_list_sorted,
        symbols_for_stocks_and_benchmarks_yf, fx_pairs_for_api_yf,
        internal_to_yf_map, yf_to_internal_map_hist, splits_by_internal_symbol,
        raw_data_cache_file, raw_data_cache_key,
        daily_results_cache_file, daily_results_cache_key,
        filter_desc
    )

def _load_or_fetch_raw_historical_data(
    symbols_to_fetch_yf: List[str], # Combined list of stock, benchmark YF tickers
    fx_pairs_to_fetch_yf: List[str], # List of YF FX tickers (e.g., ['JPY=X', 'EUR=X'])
    start_date: date,
    end_date: date,
    use_raw_data_cache: bool,
    raw_data_cache_file: str,
    raw_data_cache_key: str # Key to validate cache content
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], bool]:
    """
    Loads adjusted historical price/FX data from cache if valid, otherwise fetches
    fresh data from Yahoo Finance and updates the cache.
    FX rates fetched are OTHER CURRENCY per 1 USD (e.g., from JPY=X).

    Args:
        symbols_to_fetch_yf: List of YF stock/benchmark tickers.
        fx_pairs_to_fetch_yf: List of YF FX tickers (e.g., 'JPY=X').
        start_date: Start date for historical data.
        end_date: End date for historical data.
        use_raw_data_cache: Flag to enable caching.
        raw_data_cache_file: Path to the raw historical data cache file.
        raw_data_cache_key: Cache validation key.

    Returns:
        A tuple containing:
        - historical_prices_yf_adjusted: Dict {yf_ticker: DataFrame} for stocks/benchmarks.
        - historical_fx_yf: Dict {yf_fx_ticker: DataFrame} for FX pairs (e.g., {'JPY=X': df}).
        - fetch_failed: Boolean indicating if fetching critical data failed.
    """
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
    historical_fx_yf: Dict[str, pd.DataFrame] = {}
    cache_valid_raw = False
    fetch_failed = False # Track if essential fetching fails

    # --- 1. Try Loading Cache ---
    if use_raw_data_cache and raw_data_cache_file and os.path.exists(raw_data_cache_file):
        print(f"Hist Raw: Attempting to load raw data cache: {raw_data_cache_file}")
        try:
            with open(raw_data_cache_file, 'r') as f: cached_data = json.load(f)
            if cached_data.get('cache_key') == raw_data_cache_key:
                deserialization_errors = 0
                # Load Prices
                cached_prices = cached_data.get('historical_prices', {})
                for symbol in symbols_to_fetch_yf:
                    data_json = cached_prices.get(symbol); df = None
                    if data_json:
                        try:
                            df = pd.read_json(StringIO(data_json), orient='split', dtype={'price': float})
                            df.index = pd.to_datetime(df.index, errors='coerce').date # Convert index to date objects
                            df = df.dropna(subset=['price']) # Ensure price is valid
                            df = df[pd.notnull(df.index)] # Drop NaT indices
                            if not df.empty: historical_prices_yf_adjusted[symbol] = df.sort_index()
                            else: historical_prices_yf_adjusted[symbol] = pd.DataFrame() # Store empty if no valid data
                        except Exception as e_deser:
                            # print(f"DEBUG: Error deserializing cached price for {symbol}: {e_deser}") # Optional
                            deserialization_errors += 1; historical_prices_yf_adjusted[symbol] = pd.DataFrame()
                    else: historical_prices_yf_adjusted[symbol] = pd.DataFrame() # Symbol missing from cache

                # Load FX Rates (keyed by JPY=X, etc.)
                cached_fx = cached_data.get('historical_fx_rates', {})
                for pair in fx_pairs_to_fetch_yf:
                    data_json = cached_fx.get(pair); df = None
                    if data_json:
                        try:
                            df = pd.read_json(StringIO(data_json), orient='split', dtype={'price': float})
                            df.index = pd.to_datetime(df.index, errors='coerce').date # Convert index to date objects
                            df = df.dropna(subset=['price'])
                            df = df[pd.notnull(df.index)]
                            if not df.empty: historical_fx_yf[pair] = df.sort_index()
                            else: historical_fx_yf[pair] = pd.DataFrame()
                        except Exception as e_deser_fx:
                            # print(f"DEBUG: Error deserializing cached FX for {pair}: {e_deser_fx}") # Optional
                            deserialization_errors += 1; historical_fx_yf[pair] = pd.DataFrame()
                    else: historical_fx_yf[pair] = pd.DataFrame() # Pair missing

                # Validate Cache Completeness
                all_symbols_loaded = all(s in historical_prices_yf_adjusted for s in symbols_to_fetch_yf)
                all_fx_loaded = all(p in historical_fx_yf for p in fx_pairs_to_fetch_yf)

                if all_symbols_loaded and all_fx_loaded and deserialization_errors == 0:
                    print("Hist Raw: Cache valid and complete.")
                    cache_valid_raw = True
                else:
                    print(f"Hist WARN: RAW cache incomplete or errors ({deserialization_errors}). Will refetch if needed.")
                    # Keep partially loaded data, fetch might fill gaps
            else:
                print(f"Hist Raw: Cache key mismatch. Ignoring cache.")
        except Exception as e:
            print(f"Error reading hist RAW cache {raw_data_cache_file}: {e}. Ignoring cache.")
            historical_prices_yf_adjusted = {} # Clear potentially corrupted data
            historical_fx_yf = {}

    # --- 2. Fetch Missing Data if Cache Invalid/Incomplete ---
    if not cache_valid_raw:
        print("Hist Raw: Fetching data from Yahoo Finance...")
        # Determine which symbols *still* need fetching (if cache was partial)
        symbols_needing_fetch = [s for s in symbols_to_fetch_yf if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty]
        fx_needing_fetch = [p for p in fx_pairs_to_fetch_yf if p not in historical_fx_yf or historical_fx_yf[p].empty]

        if symbols_needing_fetch:
             print(f"Hist Raw: Fetching {len(symbols_needing_fetch)} stock/benchmark symbols...")
             fetched_stock_data = fetch_yf_historical(symbols_needing_fetch, start_date, end_date) # Assumes helper exists
             historical_prices_yf_adjusted.update(fetched_stock_data) # Add fetched data
        else:
             print("Hist Raw: All stock/benchmark symbols found in cache or not needed.")


        if fx_needing_fetch:
             print(f"Hist Raw: Fetching {len(fx_needing_fetch)} FX pairs...")
             # Use the same fetcher, assuming it handles CURRENCY=X tickers
             fetched_fx_data = fetch_yf_historical(fx_needing_fetch, start_date, end_date)
             historical_fx_yf.update(fetched_fx_data) # Add fetched data
        else:
             print("Hist Raw: All FX pairs found in cache or not needed.")


        # --- Validation after fetch ---
        final_symbols_missing = [s for s in symbols_to_fetch_yf if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty]
        final_fx_missing = [p for p in fx_pairs_to_fetch_yf if p not in historical_fx_yf or historical_fx_yf[p].empty]

        if final_symbols_missing:
             print(f"Hist WARN: Failed to fetch/load adjusted prices for: {', '.join(final_symbols_missing)}")
             # Decide if this is critical - depends if any portfolio holdings use these
             # For now, let's flag fetch_failed only if essential FX is missing
        if final_fx_missing:
             print(f"Hist WARN: Failed to fetch/load FX rates for: {', '.join(final_fx_missing)}")
             # If ANY required FX pair is missing, mark as failed for safety
             fetch_failed = True

        # --- 3. Update Cache if Fetch Occurred ---
        if use_raw_data_cache and (symbols_needing_fetch or fx_needing_fetch): # Only save if we fetched something
            print(f"Hist Raw: Saving updated raw data to cache: {raw_data_cache_file}")
            # Prepare JSON-serializable data
            prices_to_cache = {
                symbol: df.to_json(orient='split', date_format='iso')
                for symbol, df in historical_prices_yf_adjusted.items() if not df.empty # Only cache non-empty
            }
            fx_to_cache = {
                pair: df.to_json(orient='split', date_format='iso')
                for pair, df in historical_fx_yf.items() if not df.empty # Only cache non-empty
            }
            cache_content = {
                'cache_key': raw_data_cache_key,
                'timestamp': datetime.now().isoformat(),
                'historical_prices': prices_to_cache,
                'historical_fx_rates': fx_to_cache
            }
            try:
                cache_dir_raw = os.path.dirname(raw_data_cache_file)
                if cache_dir_raw: os.makedirs(cache_dir_raw, exist_ok=True)
                with open(raw_data_cache_file, 'w') as f: json.dump(cache_content, f, indent=2)
            except Exception as e: print(f"Error writing hist RAW cache: {e}")
        elif not use_raw_data_cache:
             print("Hist Raw: Caching disabled, skipping save.")

    # --- 4. Final Check and Return ---
    # Check again if critical data is missing after cache/fetch attempts
    if not fetch_failed and fx_pairs_to_fetch_yf: # Re-check FX only if pairs were needed
         if any(p not in historical_fx_yf or historical_fx_yf[p].empty for p in fx_pairs_to_fetch_yf):
             print("Hist ERROR: Critical FX data missing after final check.")
             fetch_failed = True

    if not fetch_failed and symbols_to_fetch_yf: # Re-check stocks
          if any(s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty for s in symbols_to_fetch_yf):
               # Allow proceeding but maybe warn later if specific symbols are needed and missing
               print("Hist WARN: Some stock/benchmark data missing after final check.")

    return historical_prices_yf_adjusted, historical_fx_yf, fetch_failed

def _load_or_calculate_daily_results(
    use_daily_results_cache: bool,
    daily_results_cache_file: Optional[str],
    daily_results_cache_key: Optional[str],
    start_date: date,
    end_date: date,
    # Arguments needed by the worker function & prep:
    transactions_df_effective: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame], # Keyed by 'JPY=X' etc.
    display_currency: str, # Target currency for calculations
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    clean_benchmark_symbols_yf: List[str],
    num_processes: Optional[int] = None,
    current_hist_version: str = "v10", # For logging
    filter_desc: str = "All Accounts" # For logging
) -> Tuple[pd.DataFrame, bool, str]: # Returns daily_df, cache_was_valid, status_update
    """
    Loads calculated daily results (value, flow, return, gain) from cache if valid.
    Otherwise, calculates them using parallel processing and saves to cache.
    """
    daily_df = pd.DataFrame()
    cache_valid_daily_results = False
    status_update = ""

    # --- 1. Check Daily Results Cache ---
    if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key:
        if os.path.exists(daily_results_cache_file):
            print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Found daily cache file. Checking key...")
            try:
                with open(daily_results_cache_file, 'r') as f: cached_daily_data = json.load(f)
                if cached_daily_data.get('cache_key') == daily_results_cache_key:
                    print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Cache MATCH. Loading...")
                    results_json = cached_daily_data.get('daily_results_json')
                    if results_json:
                        try:
                            daily_df = pd.read_json(StringIO(results_json), orient='split')
                            # --- Cache Validation ---
                            # Ensure index is DatetimeIndex after loading
                            if not isinstance(daily_df.index, pd.DatetimeIndex):
                                 daily_df.index = pd.to_datetime(daily_df.index, errors='coerce')
                                 daily_df = daily_df[pd.notnull(daily_df.index)] # Drop NaT

                            daily_df.sort_index(inplace=True)
                            # Check for essential columns generated by worker AND subsequent calcs
                            required_cols = ['value', 'net_flow', 'daily_return', 'daily_gain']
                            for bm_symbol in clean_benchmark_symbols_yf: required_cols.append(f"{bm_symbol} Price")
                            missing_cols = [c for c in required_cols if c not in daily_df.columns]

                            if not missing_cols and not daily_df.empty:
                                print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Loaded {len(daily_df)} rows from cache.")
                                cache_valid_daily_results = True
                                status_update = " Daily results loaded from cache."
                            else:
                                 print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Daily cache missing columns ({missing_cols}) or empty. Recalculating.")
                                 daily_df = pd.DataFrame() # Reset df
                        except Exception as e_load_df:
                             print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Error deserializing daily cache DF: {e_load_df}. Recalculating.")
                             daily_df = pd.DataFrame() # Reset df
                    else: print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Daily cache missing result data. Recalculating.")
                else: print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Daily results cache key MISMATCH. Recalculating.")
            except Exception as e_load_cache: print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Error reading daily cache: {e_load_cache}. Recalculating.")
        else: print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Daily cache file not found. Calculating.")
    elif not use_daily_results_cache:
         print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Daily results caching disabled. Calculating.")
    else: # Caching enabled but file/key invalid from prep stage
         print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Daily cache file/key invalid. Calculating.")


    # --- 2. Calculate Daily Metrics if Cache Invalid/Disabled ---
    if not cache_valid_daily_results:
        status_update = " Calculating daily values..."

        # --- Determine Calculation Dates ---
        first_tx_date = transactions_df_effective['Date'].min().date() if not transactions_df_effective.empty else start_date
        calc_start_date = max(start_date, first_tx_date)
        calc_end_date = end_date

        # Use benchmark data to find likely market days
        market_day_source_symbol = 'SPY' if 'SPY' in historical_prices_yf_adjusted else (clean_benchmark_symbols_yf[0] if clean_benchmark_symbols_yf else None)
        market_days_index = pd.Index([], dtype='object')
        if market_day_source_symbol and market_day_source_symbol in historical_prices_yf_adjusted:
            bench_df = historical_prices_yf_adjusted[market_day_source_symbol]
            if not bench_df.empty and isinstance(bench_df.index, (pd.DatetimeIndex, pd.Index)): # Check index type
                 try:
                     # Convert index to date objects robustly
                     market_days_index = pd.Index(pd.to_datetime(bench_df.index, errors='coerce').date)
                     market_days_index = market_days_index.dropna()
                 except Exception as e_idx: print(f"WARN: Failed converting benchmark index for market days: {e_idx}")

        if market_days_index.empty:
            print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): No market days found. Using business day range.");
            all_dates_to_process = pd.date_range(start=calc_start_date, end=calc_end_date, freq='B').date.tolist()
        else:
             # Filter market days index to be within the required calculation range
             all_dates_to_process = market_days_index[(market_days_index >= calc_start_date) & (market_days_index <= calc_end_date)].tolist()

        if not all_dates_to_process:
             print(f"Hist ERROR ({current_hist_version} / Scope: {filter_desc}): No calculation dates found in range.")
             return pd.DataFrame(), False, status_update + " No calculation dates found." # Return empty df

        print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Calculating {len(all_dates_to_process)} daily metrics parallel...")

        # --- Setup and Run Pool ---
        # Use functools.partial to pass fixed arguments to the worker
        partial_worker = partial(
            _calculate_daily_metrics_worker, # Assumes this worker function is defined
            transactions_df=transactions_df_effective,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            target_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            benchmark_symbols_yf=clean_benchmark_symbols_yf
        )

        daily_results_list = []
        pool_start_time = time.time()
        if num_processes is None:
            try: num_processes = max(1, os.cpu_count() - 1) # Leave one core free
            except NotImplementedError: num_processes = 1
        # Ensure num_processes is at least 1
        num_processes = max(1, num_processes)

        try:
            # Use freeze_support() if needed, especially for packaged apps (see previous discussion)
            # multiprocessing.freeze_support() # Typically called only in main script guard

            # Consider using imap_unordered for potentially better performance with uneven task times
            # Chunksize calculation aims for roughly 4 chunks per worker initially
            chunksize = max(1, len(all_dates_to_process) // (num_processes * 4))
            print(f"Hist Daily: Starting pool with {num_processes} processes, chunksize={chunksize}")

            # Use context manager for the pool
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap_unordered for potentially faster processing as results come in
                results_iterator = pool.imap_unordered(partial_worker, all_dates_to_process, chunksize=chunksize)
                # Process results as they complete
                for i, result in enumerate(results_iterator):
                    if i % 100 == 0 and i > 0: # Print progress occasionally
                         print(f"  Processed {i}/{len(all_dates_to_process)} days...")
                    if result: # Append if worker returned data (not None on error)
                        daily_results_list.append(result)
                print(f"  Finished processing all {len(all_dates_to_process)} days.")


        except Exception as e_pool:
             print(f"Hist CRITICAL ({current_hist_version} / Scope: {filter_desc}): Pool failed: {e_pool}");
             import traceback
             traceback.print_exc()
             # Cannot proceed without daily results
             return pd.DataFrame(), False, status_update + " Multiprocessing failed."
        finally: # Ensure pool timing is printed even if errors occur later
             pool_end_time = time.time()
             print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Pool finished in {pool_end_time - pool_start_time:.2f}s.")


        # --- Process Pool Results ---
        successful_results = [r for r in daily_results_list if not r.get('worker_error', False)]
        failed_count = len(all_dates_to_process) - len(successful_results)
        if failed_count > 0:
            status_update += f" ({failed_count} dates failed in worker)."
        if not successful_results:
             return pd.DataFrame(), False, status_update + " All daily calculations failed in worker."

        try:
            daily_df = pd.DataFrame(successful_results)
            daily_df['Date'] = pd.to_datetime(daily_df['Date'])
            daily_df.set_index('Date', inplace=True)
            daily_df.sort_index(inplace=True)

            # Convert columns to numeric, coercing errors
            cols_to_numeric = ['value', 'net_flow'] + [f"{bm} Price" for bm in clean_benchmark_symbols_yf if f"{bm} Price" in daily_df.columns]
            for col in cols_to_numeric:
                daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')

            # Drop rows where essential 'value' calculation failed
            initial_rows_calc = len(daily_df)
            daily_df.dropna(subset=['value'], inplace=True)
            rows_dropped = initial_rows_calc - len(daily_df)
            if rows_dropped > 0: status_update += f" ({rows_dropped} rows dropped post-calc due to NaN value)."

            if daily_df.empty:
                return pd.DataFrame(), False, status_update + " All rows dropped due to NaN portfolio value."

            # Calculate Daily Gain and Return
            previous_value = daily_df['value'].shift(1)
            # Ensure net_flow is numeric and fill NaNs with 0 for calculation
            net_flow_filled = daily_df['net_flow'].fillna(0.0)

            daily_df['daily_gain'] = daily_df['value'] - previous_value - net_flow_filled
            daily_df['daily_return'] = np.nan # Initialize

            # Calculate return where previous value is valid and non-zero
            valid_prev_value_mask = previous_value.notna() & (abs(previous_value) > 1e-9)
            daily_df.loc[valid_prev_value_mask, 'daily_return'] = daily_df.loc[valid_prev_value_mask, 'daily_gain'] / previous_value.loc[valid_prev_value_mask]

            # Handle cases where previous value was zero
            zero_gain_mask = daily_df['daily_gain'].notna() & (abs(daily_df['daily_gain']) < 1e-9)
            zero_prev_value_mask = previous_value.notna() & (abs(previous_value) <= 1e-9)
            # Set return to 0% if both gain and previous value were zero
            daily_df.loc[zero_gain_mask & zero_prev_value_mask, 'daily_return'] = 0.0
            # Handle infinite return if gain exists but previous value was zero (unlikely for portfolio value)
            # Note: This might produce inf/-inf, which is mathematically correct but maybe not desired for display

            # Set first day's gain/return to NaN
            if not daily_df.empty:
                 first_idx = daily_df.index[0]
                 daily_df.loc[first_idx, 'daily_gain'] = np.nan
                 daily_df.loc[first_idx, 'daily_return'] = np.nan

            if 'daily_gain' not in daily_df.columns or 'daily_return' not in daily_df.columns:
                 return pd.DataFrame(), False, status_update + " Failed calc daily gain/return."

            status_update += f" {len(daily_df)} days calculated."

        except Exception as e_df_create:
             print(f"Hist CRITICAL ({current_hist_version} / Scope: {filter_desc}): Failed create/process daily DF from results: {e_df_create}");
             import traceback
             traceback.print_exc()
             return pd.DataFrame(), False, status_update + " Error processing results."

        # --- 3. Save Daily Results Cache ---
        if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key and not daily_df.empty:
            print(f"Hist Daily ({current_hist_version} / Scope: {filter_desc}): Saving daily results to cache: {daily_results_cache_file}")
            cache_content = {
                'cache_key': daily_results_cache_key,
                'timestamp': datetime.now().isoformat(),
                'daily_results_json': daily_df.to_json(orient='split', date_format='iso') # Save df with calculated returns
            }
            try:
                cache_dir = os.path.dirname(daily_results_cache_file);
                if cache_dir: os.makedirs(cache_dir, exist_ok=True);
                with open(daily_results_cache_file, 'w') as f: json.dump(cache_content, f, indent=2)
            except Exception as e_save_cache:
                print(f"Hist WARN ({current_hist_version} / Scope: {filter_desc}): Error writing daily cache: {e_save_cache}")


    return daily_df, cache_valid_daily_results, status_update

def _calculate_accumulated_gains_and_resample(
    daily_df: pd.DataFrame, # Input DataFrame with daily value, gain, return, bench prices
    benchmark_symbols_yf: List[str], # Cleaned list of benchmark tickers
    interval: str # 'D', 'W', 'M'
) -> Tuple[pd.DataFrame, float, str]: # Returns final_df, twr_factor, status_update
    """
    Calculates portfolio and benchmark accumulated gains based on daily returns.
    Applies resampling (Weekly or Monthly) if specified.

    Args:
        daily_df: DataFrame indexed by Date, requires 'value', 'daily_gain',
                  'daily_return', and '{Bench} Price' columns.
        benchmark_symbols_yf: Cleaned list of benchmark YF tickers.
        interval: Resampling interval ('D', 'W', 'M').

    Returns:
        A tuple containing:
        - final_df_filtered: DataFrame with results at the specified interval.
        - final_twr_factor: The final portfolio TWR factor (cumulative product).
        - status_update: Status message snippet.
    """
    if daily_df.empty:
        return pd.DataFrame(), np.nan, " (No daily data for final calcs)"
    if 'daily_return' not in daily_df.columns:
         return pd.DataFrame(), np.nan, " (Daily return column missing)"

    status_update = ""
    final_twr_factor = np.nan
    results_df = daily_df.copy() # Work on a copy

    try:
        # --- Portfolio Accumulated Gain ---
        gain_factors_portfolio = (1 + results_df['daily_return'].fillna(0.0))
        results_df['Portfolio Accumulated Gain'] = gain_factors_portfolio.cumprod()
        # Ensure first value is NaN if daily_return was NaN
        if not results_df.empty and pd.isna(results_df['daily_return'].iloc[0]):
             results_df.iloc[0, results_df.columns.get_loc('Portfolio Accumulated Gain')] = np.nan

        # Extract TWR factor *before* resampling if interval != 'D'
        if not results_df.empty and 'Portfolio Accumulated Gain' in results_df.columns:
            last_valid_twr = results_df['Portfolio Accumulated Gain'].dropna().iloc[-1:]
            if not last_valid_twr.empty:
                 final_twr_factor = last_valid_twr.iloc[0]

        # --- Benchmark Accumulated Gain ---
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col = f"{bm_symbol} Accumulated Gain"
            if price_col in results_df.columns:
                bench_prices_no_na = results_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    # Calculate daily returns based on the adjusted prices
                    bench_daily_returns = bench_prices_no_na.pct_change()
                    # Reindex to match results_df, forward fill gaps, fill initial NaN with 0 for cumprod start
                    bench_daily_returns = bench_daily_returns.reindex(results_df.index).ffill().fillna(0.0)

                    gain_factors_bench = (1 + bench_daily_returns)
                    accum_gains_bench = gain_factors_bench.cumprod()
                    results_df[accum_col] = accum_gains_bench
                    # Set first day's accumulated gain to NaN (return relative to start)
                    if not results_df.empty:
                        results_df.iloc[0, results_df.columns.get_loc(accum_col)] = np.nan
                else: results_df[accum_col] = np.nan # No valid prices
            else: results_df[accum_col] = np.nan # Price column missing
        status_update += " Accum gain calc complete."

        # --- Select Final Columns ---
        columns_to_keep = ['value', 'daily_gain', 'daily_return', 'Portfolio Accumulated Gain']
        for bm_symbol in benchmark_symbols_yf:
            if f"{bm_symbol} Price" in results_df.columns: columns_to_keep.append(f"{bm_symbol} Price")
            if f"{bm_symbol} Accumulated Gain" in results_df.columns: columns_to_keep.append(f"{bm_symbol} Accumulated Gain")
        columns_to_keep = [col for col in columns_to_keep if col in results_df.columns] # Ensure they exist
        final_df_filtered = results_df[columns_to_keep].copy()
        # Rename columns for final output
        final_df_filtered.rename(columns={'value': 'Portfolio Value', 'daily_gain': 'Portfolio Daily Gain'}, inplace=True)


        # --- Apply Resampling ---
        if interval != 'D' and not final_df_filtered.empty:
             print(f"Hist Final: Resampling to interval '{interval}'...")
             try:
                 resample_freq = interval
                 # Define aggregation methods
                 resampling_agg = {
                     'Portfolio Value': 'last', # Take last value in period
                     'Portfolio Daily Gain': 'sum', # Sum gains over period
                 }
                 # Benchmark prices: take last
                 for bm_symbol in benchmark_symbols_yf:
                      price_col = f"{bm_symbol} Price"
                      if price_col in final_df_filtered.columns:
                          resampling_agg[price_col] = 'last'

                 # Resample essential columns first
                 resampled_essentials = final_df_filtered.resample(resample_freq).agg(resampling_agg)

                 # Recalculate Accumulated Gains on the resampled data
                 if 'Portfolio Value' in resampled_essentials.columns:
                     # Calculate returns based on resampled (e.g., weekly/monthly) values
                     resampled_returns = resampled_essentials['Portfolio Value'].pct_change().fillna(0.0)
                     resampled_essentials['Portfolio Accumulated Gain'] = (1 + resampled_returns).cumprod()
                     # Set first resampled gain to NaN
                     if not resampled_essentials.empty:
                          resampled_essentials.iloc[0, resampled_essentials.columns.get_loc('Portfolio Accumulated Gain')] = np.nan
                     # Update TWR factor based on resampled data if needed (though daily usually preferred)
                     # last_resampled_twr = resampled_essentials['Portfolio Accumulated Gain'].dropna().iloc[-1:]
                     # if not last_resampled_twr.empty: final_twr_factor = last_resampled_twr.iloc[0]
                 else: resampled_essentials['Portfolio Accumulated Gain'] = np.nan

                 for bm_symbol in benchmark_symbols_yf:
                      price_col = f"{bm_symbol} Price"; accum_col = f"{bm_symbol} Accumulated Gain"
                      if price_col in resampled_essentials.columns:
                          resampled_bench_returns = resampled_essentials[price_col].pct_change().fillna(0.0)
                          resampled_essentials[accum_col] = (1 + resampled_bench_returns).cumprod()
                          if not resampled_essentials.empty:
                               resampled_essentials.iloc[0, resampled_essentials.columns.get_loc(accum_col)] = np.nan
                      else: resampled_essentials[accum_col] = np.nan

                 # Assign resampled data back
                 final_df_filtered = resampled_essentials
                 status_update += f" Resampled to '{interval}'."

             except Exception as e_resample:
                  print(f"Hist WARN: Failed resampling to interval '{interval}': {e_resample}. Returning daily results.")
                  status_update += f" Resampling failed ('{interval}')."
                  # Keep final_df_filtered as the daily results

    except Exception as e_accum:
        print(f"Hist CRITICAL: Accum gain/resample calc error: {e_accum}");
        import traceback
        traceback.print_exc()
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update # Return empty on critical error


    return final_df_filtered, final_twr_factor, status_update
    
# --- Historical Performance Calculation Wrapper Function (Refactored for Parallelism + Results Cache) ---
def calculate_historical_performance(
    transactions_csv_file: str,
    start_date: date,
    end_date: date,
    interval: str, # 'D', 'W', 'M'
    benchmark_symbols_yf: List[str], # Raw list from caller
    display_currency: str,
    account_currency_map: Dict,
    default_currency: str,
    use_raw_data_cache: bool = True,
    use_daily_results_cache: bool = True,
    num_processes: Optional[int] = None,
    include_accounts: Optional[List[str]] = None,
    exclude_accounts: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Calculates historical portfolio performance (TWR-like accumulated gain)
    and benchmark performance over a date range using parallel daily calculations.
    Filters calculations based on include/exclude accounts. Orchestrates calls
    to helper functions for different stages.

    (Keep Args and Returns documentation as before)
    """
    CURRENT_HIST_VERSION = "v10" # Or manage this globally
    start_time_hist = time.time()

    # --- Initial Checks & Cleaning of Inputs ---
    if not YFINANCE_AVAILABLE: return pd.DataFrame(), "Error: yfinance library not installed."
    if start_date >= end_date: return pd.DataFrame(), "Error: Start date must be before end date."
    if interval not in ['D', 'W', 'M']: return pd.DataFrame(), f"Error: Invalid interval '{interval}'."

    # Clean the benchmark symbols list passed in
    clean_benchmark_symbols_yf = []
    if benchmark_symbols_yf and isinstance(benchmark_symbols_yf, list):
        clean_benchmark_symbols_yf = [
            b.upper().strip() for b in benchmark_symbols_yf if isinstance(b, str) and b.strip()
        ]
    elif not benchmark_symbols_yf:
        print(f"Hist INFO ({CURRENT_HIST_VERSION}): No benchmark symbols provided.")
    else:
        print(f"Hist WARN ({CURRENT_HIST_VERSION}): Invalid benchmark_symbols_yf type provided: {type(benchmark_symbols_yf)}. Ignoring benchmarks.")

    # --- 1. Prepare Inputs (Load, Clean, Filter, Symbols, Keys, Splits) ---
    (transactions_df_effective, original_transactions_df, ignored_indices, ignored_reasons,
     all_available_accounts_list, included_accounts_list_sorted, excluded_accounts_list_sorted,
     symbols_for_stocks_and_benchmarks_yf, fx_pairs_for_api_yf, # Use the correct FX tickers list YF needs (e.g., JPY=X)
     internal_to_yf_map, yf_to_internal_map_hist, splits_by_internal_symbol,
     raw_data_cache_file, raw_data_cache_key,
     daily_results_cache_file, daily_results_cache_key,
     filter_desc) = _prepare_historical_inputs(
        transactions_csv_file, account_currency_map, default_currency,
        include_accounts, exclude_accounts, start_date, end_date,
        clean_benchmark_symbols_yf, # Pass the cleaned list
        display_currency,
        current_hist_version=CURRENT_HIST_VERSION,
        raw_cache_prefix=HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        daily_cache_prefix=DAILY_RESULTS_CACHE_PATH_PREFIX
    )

    # --- Handle early exit if inputs preparation failed ---
    if transactions_df_effective is None:
         status_msg = f"Error: Failed to load/clean transactions from '{transactions_csv_file}' during preparation."
         # Prepare ignored_df if possible
         ignored_df = pd.DataFrame()
         if ignored_indices and original_transactions_df is not None:
              valid_indices = sorted([idx for idx in ignored_indices if idx in original_transactions_df.index])
              if valid_indices: ignored_df = original_transactions_df.loc[valid_indices].copy()
              # Optionally add reasons if ignored_reasons is populated
         # Consider what to return for other values if needed by caller (though unlikely on error)
         return pd.DataFrame(), status_msg # Return empty DataFrame on critical input error

    # Initialize status message and other variables
    status_msg = f"Historical ({CURRENT_HIST_VERSION} / Scope: {filter_desc}): Inputs prepared."
    if ignored_reasons: status_msg += f" ({len(ignored_reasons)} ignored)."
    processed_warnings = set()
    final_twr_factor = np.nan
    daily_df = pd.DataFrame() # Initialize daily results dataframe


    # --- 3. Load or Fetch ADJUSTED Historical Raw Data ---
    historical_prices_yf_adjusted, historical_fx_yf, fetch_failed = _load_or_fetch_raw_historical_data(
        symbols_to_fetch_yf=symbols_for_stocks_and_benchmarks_yf, # Use the correct list from prep
        fx_pairs_to_fetch_yf=fx_pairs_for_api_yf, # Use the correct FX ticker list from prep
        start_date=start_date,
        end_date=end_date,
        use_raw_data_cache=use_raw_data_cache,
        raw_data_cache_file=raw_data_cache_file, # From prep
        raw_data_cache_key=raw_data_cache_key # From prep
    )

    # --- Check for Fetch Failure ---
    if fetch_failed:
         status_msg += " Error: Failed fetching critical historical FX/Price data."
         # Return empty DataFrame as calculation cannot proceed reliably
         return pd.DataFrame(), status_msg

    status_msg += " Raw adjusted data loaded/fetched."


    # --- 4. Derive Unadjusted Prices ---
    # Uses the existing helper function _unadjust_prices
    print("Deriving unadjusted prices using split data...") # Add print statement
    historical_prices_yf_unadjusted = _unadjust_prices(
        adjusted_prices_yf=historical_prices_yf_adjusted, # Input the adjusted prices
        yf_to_internal_map=yf_to_internal_map_hist,     # Input the correct map
        splits_by_internal_symbol=splits_by_internal_symbol, # Input splits dict
        processed_warnings=processed_warnings              # Pass the set to track warnings
    )
    status_msg += " Unadjusted prices derived."


    # --- 5 & 6. Load or Calculate Daily Results ---
    daily_df, cache_valid_daily_results, status_update = _load_or_calculate_daily_results(
        use_daily_results_cache=use_daily_results_cache,
        daily_results_cache_file=daily_results_cache_file, # from prep
        daily_results_cache_key=daily_results_cache_key, # from prep
        start_date=start_date, # Original start date
        end_date=end_date, # Original end date
        transactions_df_effective=transactions_df_effective, # from prep
        historical_prices_yf_unadjusted=historical_prices_yf_unadjusted, # from step 4
        historical_prices_yf_adjusted=historical_prices_yf_adjusted, # from step 3
        historical_fx_yf=historical_fx_yf, # from step 3
        display_currency=display_currency,
        internal_to_yf_map=internal_to_yf_map, # from prep
        account_currency_map=account_currency_map, # Original arg
        default_currency=default_currency, # Original arg
        clean_benchmark_symbols_yf=clean_benchmark_symbols_yf, # Cleaned list
        num_processes=num_processes, # Original arg
        current_hist_version=CURRENT_HIST_VERSION,
        filter_desc=filter_desc # from prep
    )
    status_msg += status_update # Append status from helper

    # --- Check if daily calculation failed ---
    if daily_df.empty: # Check if DF is empty after call
        # Status message should already contain reason if empty
        return pd.DataFrame(), status_msg

    if daily_df.empty: # Check if DF is empty after call
        return pd.DataFrame(), status_msg

    # --- 7. Calculate Accumulated Gains and Resample ---
    final_df_filtered, final_twr_factor, status_update_final = _calculate_accumulated_gains_and_resample(
        daily_df=daily_df, # Pass the calculated daily results
        benchmark_symbols_yf=clean_benchmark_symbols_yf, # Pass cleaned benchmarks
        interval=interval # Pass resampling interval
    )
    status_msg += status_update_final

    # --- 8. Final Status and Return ---
    end_time_hist = time.time()
    print(f"--- Historical Performance Calculation Finished ({CURRENT_HIST_VERSION} / Scope: {filter_desc}) ---")
    print(f"Total Historical Calc Time: {end_time_hist - start_time_hist:.2f} seconds")

    # Determine final status based on messages/warnings/errors accumulated
    final_status = status_msg # Use accumulated status
    if 'failed' in status_msg.lower() or 'error' in status_msg.lower(): pass
    elif final_df_filtered.empty: final_status += " Success (Empty Result)."
    else: final_status += " Success."
    # Append TWR factor (now returned by the helper)
    final_status += f"|||TWR_FACTOR:{final_twr_factor:.6f}" if pd.notna(final_twr_factor) else "|||TWR_FACTOR:NaN"

    # Final column ordering (optional, good practice)
    if not final_df_filtered.empty:
        final_cols_order = ['Portfolio Value', 'Portfolio Daily Gain', 'daily_return', 'Portfolio Accumulated Gain']
        for bm in clean_benchmark_symbols_yf:
            if f"{bm} Price" in final_df_filtered.columns: final_cols_order.append(f"{bm} Price")
            if f"{bm} Accumulated Gain" in final_df_filtered.columns: final_cols_order.append(f"{bm} Accumulated Gain")
        final_cols_order = [c for c in final_cols_order if c in final_df_filtered.columns] # Ensure columns exist
        try:
             final_df_filtered = final_df_filtered[final_cols_order]
        except KeyError:
             print("Warning: Could not reorder final columns.") # Should not happen if list comprehension is correct

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
        # "SET_Only": {"include": ['SET'], "exclude": None},
        "IBKR_E*TRADE": {"include": test_accounts_subset1, "exclude": None},
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


# --- END OF FILE portfolio_logic.py ---
