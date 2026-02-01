from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd
import logging
import os
import json
import time
import sqlite3
from datetime import datetime, date, time as dt_time



import shutil

from server.dependencies import get_transaction_data, get_config_manager, reload_data, get_global_db_connection, get_user_db_connection
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
from utils_time import get_est_today, get_latest_trading_date
from portfolio_analyzer import (
    calculate_periodic_returns, 
    extract_realized_capital_gains_history, 
    extract_dividend_history,
    generate_cash_interest_events
)
from market_data import MarketDataProvider, map_to_yf_symbol
from db_utils import (
    add_transaction_to_db,
    update_transaction_in_db,
    delete_transaction_from_db,
    get_database_path,
    get_db_connection,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    update_transaction_in_db,
    get_all_watchlists,
    create_watchlist,
    rename_watchlist,
    delete_watchlist,
    update_intrinsic_value_in_cache,
    upsert_screener_results,
    get_cached_screener_results
)

from risk_metrics import calculate_all_risk_metrics, calculate_drawdown_series
import config
from ibkr_connector import IBKRConnector
from config import YFINANCE_INDEX_TICKER_MAP, BENCHMARK_MAPPING
from config_manager import ConfigManager
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import numpy as np # Ensure numpy is imported
import traceback

# --- Financial Ratios Integration ---
try:
    from financial_ratios import (
        calculate_key_ratios_timeseries,
        calculate_current_valuation_ratios,
        get_comprehensive_intrinsic_value,
        get_intrinsic_value_for_symbol
    )
    FINANCIAL_RATIOS_AVAILABLE = True
except ImportError:
    logging.warning("financial_ratios.py not found or import failed. Ratios will be disabled.")
    FINANCIAL_RATIOS_AVAILABLE = False

from server.ai_analyzer import generate_stock_review
from server.screener_service import screen_stocks
from server.auth import (
    Token, User, create_access_token, get_password_hash, verify_password
)
from server.dependencies import get_current_user
from datetime import timedelta
from pydantic import BaseModel

import logging

router = APIRouter()


current_file_path = os.path.abspath(__file__)
src_server_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(src_server_dir)
project_root = os.path.dirname(src_dir)

# ... (existing code)

# Global Cache for Portfolio Summary Calculations to avoid redundant processing per-request
_PORTFOLIO_SUMMARY_CACHE = {}
_MARKET_HISTORY_CACHE = {}

def reload_data_and_clear_cache():
    """Helper to clear both transaction data cache, portfolio summary cache, and market history cache."""
    reload_data()
    _PORTFOLIO_SUMMARY_CACHE.clear()
    _MARKET_HISTORY_CACHE.clear()
    logging.info("Transaction, Summary, and Market History caches cleared.")


# Global Market Data Provider to share DB connections and cache across requests
_MDP_INSTANCE = None

def get_mdp():
    from market_data import get_shared_mdp
    return get_shared_mdp()

# --- Auth Routes ---

class UserCreate(BaseModel):
    username: str
    password: str

class UserPasswordUpdate(BaseModel):
    current_password: str
    new_password: str

from fastapi.security import OAuth2PasswordRequestForm

@router.post("/auth/register", response_model=User)
async def register(user: UserCreate, conn: sqlite3.Connection = Depends(get_global_db_connection)):
    # conn obtained from dependency is for GLOBAL DB (Users)
    
    try:
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (user.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already registered")
        
        # Hash password
        hashed_pw = get_password_hash(user.password)
        created_at = datetime.now().isoformat()
        
        cursor.execute(
            "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
            (user.username, hashed_pw, created_at)
        )
        new_user_id = cursor.lastrowid
        conn.commit()
        
        # --- Initialize User Isolation ---
        # Create user directory and initialize their portfolio DB
        user_data_dir = os.path.join(config.get_app_data_dir(), "users", user.username)
        try:
             os.makedirs(user_data_dir, exist_ok=True)
             
             # Initialize Portfolio DB
             user_db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
             from db_utils import initialize_database
             
             # We initialize the DB (creates tables)
             user_conn = initialize_database(user_db_path)
             if user_conn:
                 user_conn.close()
                 
             logging.info(f"Initialized isolated environment for user {user.username}")
             
        except Exception as e:
             # Rollback user creation if environment setup fails?
             # Ideally yes, but global DB commit already happened.
             # We log critical error.
             logging.error(f"Failed to initialize user environment for {user.username}: {e}")
             # Proceed? Or fail? The user exists but has no DB.
             # Let's try to fail harder or just logging.
             # Re-raising might be better so user knows it failed.
             raise HTTPException(status_code=500, detail="Failed to initialize user data environment")

        return User(id=new_user_id, username=user.username, is_active=True, created_at=created_at)
        
    except HTTPException:
        raise
    except Exception as e:
        # conn via dependency is closed by dependency, but we can try rollback if active transaction
        # But usually exception triggers 500.
        logging.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), conn: sqlite3.Connection = Depends(get_global_db_connection)):
    # conn is GLOBAL DB
    
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (form_data.username,))
    row = cursor.fetchone()
    
    if not row or not verify_password(form_data.password, row[2]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": row[1], "id": row[0]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.delete("/auth/me")
async def delete_user_me(
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
):
    try:
        # 1. Delete user from GLOBAL DB
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (current_user.id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()
        
        # 2. Delete user data directory
        user_data_dir = os.path.join(config.get_app_data_dir(), "users", current_user.username)
        if os.path.exists(user_data_dir):
            try:
                shutil.rmtree(user_data_dir)
                logging.info(f"Deleted data directory for user {current_user.username}")
            except Exception as e:
                logging.error(f"Failed to delete data directory for {current_user.username}: {e}")
                # We continue as the user is effectively deleted from the system
        
        return {"status": "success", "message": f"User {current_user.username} deleted"}
        
    except Exception as e:
        logging.error(f"Error deleting user {current_user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete user")

# --- End Auth Routes ---

@router.post("/auth/change-password")
async def change_password(
    password_data: UserPasswordUpdate,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user.id,))
        row = cursor.fetchone()
        
        if not row:
             raise HTTPException(status_code=404, detail="User not found")
             
        stored_hash = row[0]
        
        if not verify_password(password_data.current_password, stored_hash):
            raise HTTPException(status_code=400, detail="Incorrect current password")
            
        hashed_new_pw = get_password_hash(password_data.new_password)
        
        cursor.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed_new_pw, current_user.id))
        conn.commit()
        
        return {"status": "success", "message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error changing password for {current_user.username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update password")




@router.get("/asset_change")
async def get_asset_change(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns periodic asset change data (Annual, Monthly, Weekly, Daily).

    Args:
        currency (str): The display currency (e.g., USD, THB).
        accounts (List[str], optional): List of account names to include.
        benchmarks (List[str], optional): List of benchmark names or symbols.
        data (tuple): Dependency injection for transaction data.

    Returns:
        Dict[str, List[Dict]]: Dictionary mapping periods (Annual, etc.) to lists of asset change records.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _ # Ignore db_mtime
    ) = data

    if df.empty:
        return {}

    try:
        # Map benchmark display names to tickers
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)

        # 1. Calculate full history (using 'all' period)
        daily_df, _, _, final_status_str = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=date(2000, 1, 1), # All history
            end_date=date.today(),
            display_currency=currency,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            include_accounts=accounts,
            benchmark_symbols_yf=mapped_benchmarks,
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            interval="D",
            original_csv_file_path=original_csv_path
        )
        
        if daily_df is None or daily_df.empty:
            return {}

        # 2. Calculate periodic returns
        # Use mapped benchmarks (tickers) because daily_df has ticker columns
        periodic_returns = calculate_periodic_returns(daily_df, mapped_benchmarks)
        
        # Rename columns back to display names for the frontend
        ticker_to_name = {v: k for k, v in config.BENCHMARK_MAPPING.items()}
        for interval, p_df in periodic_returns.items():
            new_columns = []
            suffix = f" {interval}-Return"
            for col in p_df.columns:
                if col.endswith(suffix):
                    ticker_part = col[:-len(suffix)]
                    if ticker_part in ticker_to_name:
                        new_columns.append(f"{ticker_to_name[ticker_part]}{suffix}")
                    else:
                        new_columns.append(col)
                else:
                    new_columns.append(col)
            p_df.columns = new_columns
        
        # 3. Convert DataFrames to JSON-friendly dicts
        result = {}
        for period, p_df in periodic_returns.items():
            if not p_df.empty:
                # Reset index to include the date/period in the records
                # FIX: Ensure index is named 'Date' before resetting so it doesn't default to 'index'
                if p_df.index.name is None:
                    p_df.index.name = 'Date'
                
                p_df_reset = p_df.reset_index()
                
                # Convert dates to strings and handle 'index' fallback if needed
                if 'Date' in p_df_reset.columns:
                    p_df_reset['Date'] = p_df_reset['Date'].astype(str)
                elif 'index' in p_df_reset.columns:
                    # Fallback rename if for some reason it's still called 'index'
                    p_df_reset.rename(columns={'index': 'Date'}, inplace=True)
                    p_df_reset['Date'] = p_df_reset['Date'].astype(str)
                
                result[period] = clean_nans(p_df_reset.to_dict(orient="records"))
            else:
                result[period] = []
                
        return result

    except Exception as e:
        logging.error(f"Error calculating asset change: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

import config
from datetime import date, timedelta

import math

logging.info("API MODULE LOADED - VERSION 2")

def clean_nans(obj):
    """Recursively replace NaN/Infinity with None for JSON serialization."""
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj

async def _calculate_portfolio_summary_internal(
    currency: str = "USD",
    include_accounts: Optional[List[str]] = None,
    show_closed_positions: bool = True,
    data: tuple = None,
    current_user: User = None
) -> Dict[str, Any]:
    """Internal helper to calculate portfolio summary data."""
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        db_path,
        db_mtime
    ) = data

    if df.empty:
        return {"metrics": {}, "rows": []}

    # Extract interest settings - RELOAD FRESH TO AVOID STALE CACHE
    # Extract interest settings - RELOAD FRESH TO AVOID STALE CACHE
    if current_user:
        config_manager = get_config_manager(current_user)
    else:
        # Fallback if no user passed (should not happen in protected routes)
        # We rely on default behavior of get_config_manager ONLY if called via dependency injection context which is not here.
        # So we must raise error or handle it.
        # But wait, get_config_manager default IS Depends(get_current_user).
        # We cannot call it without arguments here.
        logging.error("Missing current_user in _calculate_portfolio_summary_internal")
        # Attempt to recover or fail?
        # Let's try to get it from data if possible? No.
        # We will assume callers always pass it. But for safety, we return empty config if missing.
        account_interest_rates = {}
        interest_free_thresholds = {}
        config_manager = None
    
    if config_manager:
        config_manager.load_manual_overrides()
        account_interest_rates = config_manager.manual_overrides.get("account_interest_rates", {})
        interest_free_thresholds = config_manager.manual_overrides.get("interest_free_thresholds", {})

    # --- Caching Logic ---
    # Create a unique key for this request configuration + data state
    accounts_key = tuple(sorted(include_accounts)) if include_accounts else "ALL"
    
    # ADDED: Time-based invalidation (bucketed to 5 seconds to allow fast retries)
    time_key = int(time.time() / 5)
    
    cache_key = (
        currency,
        accounts_key,
        show_closed_positions,
        db_path,
        db_mtime,
        time_key
    )
    
    if cache_key in _PORTFOLIO_SUMMARY_CACHE:
        logging.info(f"Using cached portfolio summary for key: {cache_key[:3]}...") # Partial log for brevity
        return _PORTFOLIO_SUMMARY_CACHE[cache_key]
        
    logging.info(f"Summary Cache Miss. Calculating summary. Time key: {time_key}")

    mdp = get_mdp()
    (
        overall_summary_metrics,
        summary_df,
        holdings_dict,
        account_level_metrics,
        _,
        _,
        _
    ) = calculate_portfolio_summary(
        all_transactions_df_cleaned=df,
        original_transactions_df_for_ignored=df,
        ignored_indices_from_load=set(),
        ignored_reasons_from_load={},
        fmp_api_key=getattr(config, "FMP_API_KEY", None),
        display_currency=currency,
        show_closed_positions=show_closed_positions,
        manual_overrides_dict=manual_overrides,
        user_symbol_map=user_symbol_map,
        user_excluded_symbols=user_excluded_symbols,
        include_accounts=include_accounts,
        account_currency_map=account_currency_map,
        default_currency=config.DEFAULT_CURRENCY,
        market_provider=mdp,
        account_interest_rates=account_interest_rates,
        interest_free_thresholds=interest_free_thresholds
    )
    
    if overall_summary_metrics:
        # Base calculator now correctly includes cash interest (due to fresh config load).
        # We verified 'est_annual_income_display' matches user expectations (~$5237).
        pass


        logging.info(f"Summary calculated. Total Value: {overall_summary_metrics.get('market_value')}, Day Change: {overall_summary_metrics.get('day_change_display')}")
    else:
        logging.error("Summary calculation returned None metrics")

    result = {
        "metrics": overall_summary_metrics,
        "summary_df": summary_df,
        "holdings_dict": holdings_dict,
        "account_metrics": account_level_metrics
    }
    
    # Store in cache
    _PORTFOLIO_SUMMARY_CACHE[cache_key] = result
    
    # Optional: Simple cache eviction policy (e.g. keep only last 20 entries to prevent overflow)
    if len(_PORTFOLIO_SUMMARY_CACHE) > 20:
        # Remove random or oldest item (simplest is popitem which removes LIFO in < 3.7 but FIFO in 3.7+)
        # For a dict, popitem(last=False) is not available. standard dict preserves insertion order.
        # So we can just remove the first key.
        first_key = next(iter(_PORTFOLIO_SUMMARY_CACHE))
        del _PORTFOLIO_SUMMARY_CACHE[first_key]

    return result

@router.get("/summary")
async def get_portfolio_summary(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the high-level portfolio summary (Total Value, G/L, etc.).

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names to filter by.
        data (tuple): Dependency injection for transaction data.

    Returns:
        Dict[str, Any]: A dictionary containing 'metrics' (totals) and 'account_metrics' (per-account breakdowns).
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _
    ) = data
    
    if df.empty:
        return {"error": "No transaction data available"}

    try:
        # Calculate portfolio summary using helper
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            data=data,
            current_user=current_user
        )
        
        overall_summary_metrics = summary_data["metrics"]
        account_level_metrics = summary_data["account_metrics"]

        # --- Calculate Annualized TWR ---
        annualized_twr = None
        if not df.empty:
            try:
                # Determine date range for "all" time
                min_date = df["Date"].min().date()
                max_date = date.today()
                
                # Fetch full history to calculate TWR over the entire period
                daily_df, _, _, _ = calculate_historical_performance(
                    all_transactions_df_cleaned=df,
                    original_transactions_df_for_ignored=df,
                    ignored_indices_from_load=set(),
                    ignored_reasons_from_load={},
                    start_date=min_date,
                    end_date=max_date,
                    interval="D",
                    benchmark_symbols_yf=[], # No benchmarks needed for just portfolio TWR
                    display_currency=currency,
                    account_currency_map=account_currency_map,
                    default_currency=config.DEFAULT_CURRENCY,
                    include_accounts=accounts,
                    manual_overrides_dict=manual_overrides,
                    user_symbol_map=user_symbol_map,
                    user_excluded_symbols=user_excluded_symbols,
                    original_csv_file_path=original_csv_path
                )
                
                if daily_df is not None and not daily_df.empty:
                    twr_col = "Portfolio Accumulated Gain"
                    if twr_col in daily_df.columns:
                        # Get the final TWR factor (cumulative)
                        # daily_df[twr_col] contains the cumulative TWR factor (e.g., 1.5 for 50% gain)
                        final_twr_factor = daily_df[twr_col].iloc[-1]
                        
                        # Ensure we annualize over the full REQUESTED period (from first transaction to today)
                        # This is more robust than using daily_df.index which might be truncated if data is sparse.
                        days = (max_date - min_date).days
                        
                        if days > 0 and pd.notna(final_twr_factor) and final_twr_factor > 0:
                            # Annualize: (Total_Factor)^(365.25 / Days) - 1
                            annualized_factor = final_twr_factor ** (365.25 / days)
                            annualized_twr = (annualized_factor - 1) * 100.0
            except Exception as e_twr:
                logging.warning(f"Failed to calculate Annualized TWR: {e_twr}")

        if overall_summary_metrics:
            overall_summary_metrics["annualized_twr"] = annualized_twr

            # --- Fetch Market Indices ---
            try:
                mdp = get_mdp()
                indices_data = mdp.get_index_quotes(config.INDICES_FOR_HEADER)
                overall_summary_metrics["indices"] = indices_data
            except Exception as e_indices:
                logging.warning(f"Failed to fetch market indices: {e_indices}")

        
        # Serialize DataFrame and holdings_dict keys for JSON response
        summary_df_raw = summary_data.get("summary_df")
        holdings_dict_raw = summary_data.get("holdings_dict", {})
        
        serialized_df = []
        if isinstance(summary_df_raw, pd.DataFrame):
             # Handle NaNs
             summary_df_clean = summary_df_raw.where(pd.notnull(summary_df_raw), None)
             serialized_df = summary_df_clean.to_dict(orient="records")
        
        safe_holdings_dict = {}
        if isinstance(holdings_dict_raw, dict):
             safe_holdings_dict = {f"{sym}|{acc}": val for (sym, acc), val in holdings_dict_raw.items()}

        response_data = {
            "metrics": overall_summary_metrics,
            "account_metrics": account_level_metrics,
            "summary_df": serialized_df,
            "holdings_dict": safe_holdings_dict
        }
        return clean_nans(response_data)
    except Exception as e:
        logging.error(f"Error calculating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market_history")
async def get_market_history(
    benchmarks: List[str] = Query(...),
    period: str = "1y",
    interval: str = "1d",
    currency: str = "USD",
):
    """
    Returns historical return % for given market indices/benchmarks.
    """
    # 0. Global Cache Check
    cache_key = (tuple(sorted(benchmarks)), period, interval, currency)
    now_ts_cache = time.time()
    if cache_key in _MARKET_HISTORY_CACHE:
        entry, expiry = _MARKET_HISTORY_CACHE[cache_key]
        if now_ts_cache < expiry:
            logging.info(f"Market History Cache HIT: {cache_key}")
            return entry

    try:
        from utils_time import get_est_today, get_latest_trading_date
        
        # MAPPING: Convert benchmark display names to YF tickers (reuse logic from history)
        mapped_benchmarks = []
        ticker_to_name = {}
        bm_mapping_lower = {k.lower(): v for k, v in config.BENCHMARK_MAPPING.items()}
        yf_map_lower = {k.lower(): v for k, v in config.YFINANCE_INDEX_TICKER_MAP.items()}
        
        for b in benchmarks:
            b_lower = b.lower()
            if b_lower in bm_mapping_lower:
                ticker = bm_mapping_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            elif b_lower in yf_map_lower:
                ticker = yf_map_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            else:
                mapped_benchmarks.append(b)
        
        # Determine date range (simplified logic from history)
        end_date = get_est_today() + timedelta(days=1)
        if period == "1d":
            interval = "2m" # Force Intraday
            start_date = get_latest_trading_date()
        elif period == "5d" or period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "3y":
            start_date = end_date - timedelta(days=365 * 3)
        elif period == "5y":
            start_date = end_date - timedelta(days=365 * 5)
        elif period == "10y":
            start_date = end_date - timedelta(days=365 * 10)
        elif period == "ytd":
            start_date = date(end_date.year, 1, 1)
        elif period == "all" or period == "max":
            start_date = date(1980, 1, 1) # Return full history
        else:
            start_date = end_date - timedelta(days=365)

        mdp = get_mdp()
        hist_data, _ = mdp.get_historical_data(
            symbols_yf=mapped_benchmarks,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not hist_data:
            return []

        # Process and normalize data (Return %) using Vectorized Pandas logic
        # Result should be a list of dicts: [{date: '...', '^GSPC': 0.1, ...}, ...]
        dfs = []
        for ticker in mapped_benchmarks:
            if ticker in hist_data and not hist_data[ticker].empty:
                df = hist_data[ticker][['price']].copy()
                # Normalize returns relative to first point
                first_price = df['price'].iloc[0]
                display_name = ticker_to_name.get(ticker, ticker)
                if first_price != 0:
                    df[display_name] = (df['price'] / first_price - 1) * 100
                else:
                    df[display_name] = 0.0
                
                # Keep the price column but rename it so it doesn't conflict
                df[f"{display_name}_price"] = df['price']
                
                dfs.append(df.drop(columns=['price']))
        
        if not dfs:
            return []
            
        # Combine all indices into one DataFrame
        combined_df = pd.concat(dfs, axis=1)
        combined_df = combined_df.sort_index()
        
        # Reset index to get dates as a column
        combined_df.index.name = 'date'
        combined_df = combined_df.reset_index()
        
        # Convert dates to string (using pd.to_datetime to be safe if types differ)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        is_intraday_local = any(x in interval for x in ["m", "h"])
        if is_intraday_local:
             combined_df['date'] = combined_df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
             combined_df['date'] = combined_df['date'].dt.strftime("%Y-%m-%d")

        result = clean_nans(combined_df.to_dict(orient="records"))
        
        # Cache the result for 15 minutes
        _MARKET_HISTORY_CACHE[cache_key] = (result, now_ts_cache + 900)
        
        return result

    except Exception as e:
        logging.error(f"Error in get_market_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logging.error(f"Error in get_market_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlation")
async def get_correlation_matrix(
    period: str = "1y",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the correlation matrix for the current portfolio holdings.
    
    Args:
        period (str): Lookback period (e.g., '6m', '1y', '3y').
        accounts (List[str], optional): Filter holdings by account.
        data (tuple): Dependency injection.
        
    Returns:
        Dict: structure suitable for heatmap visualization.
        {
            "assets": ["AAPL", "MSFT", ...],
            "correlation": [
                {"x": "AAPL", "y": "MSFT", "value": 0.85},
                ...
            ]
        }
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        _,
        _
    ) = data

    if df.empty:
        return {"assets": [], "correlation": []}

    try:
        # 1. Identify current holdings
        # We reuse the summary calculation to get the current list of held assets
        summary_data = await _calculate_portfolio_summary_internal(
            # Currency irrelevant for symbol identification, but required by func
            currency="USD", 
            include_accounts=accounts,
            show_closed_positions=False,
            data=data,
            current_user=current_user
        )
        
        holdings_dict = summary_data.get("holdings_dict", {})
        
        # Extract unique symbols currently held
        # holdings_dict keys are (Symbol, Account)
        held_symbols = set()
        for (sym, acct) in holdings_dict.keys():
            held_symbols.add(sym)
            
        if not held_symbols:
            return {"assets": [], "correlation": []}

        # 2. Map to YF symbols for fetching
        yf_symbols_map = {} # Internal -> YF
        yf_symbols_to_fetch = []
        
        from finutils import is_cash_symbol
        
        for sym in held_symbols:
            if is_cash_symbol(sym):
                continue
                
            yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols)
            if yf_sym:
                yf_symbols_map[sym] = yf_sym
                yf_symbols_to_fetch.append(yf_sym)
        
        if not yf_symbols_to_fetch:
             return {"assets": [], "correlation": []}

        # 3. Determine Date Range
        end_date = date.today()
        if period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "3y":
            start_date = end_date - timedelta(days=365*3)
        elif period == "5y":
            start_date = end_date - timedelta(days=365*5)
        else: # Default 1y
            start_date = end_date - timedelta(days=365)
            
        # 4. Fetch Historical Data
        mdp = get_mdp()
        hist_data, _ = mdp.get_historical_data(
            symbols_yf=yf_symbols_to_fetch,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        # 5. Build Combined DataFrame containing Prices
        # We want a DataFrame where columns = Internal Symbols (for display)
        # But we fetched by YF Symbol.
        
        price_series_dict = {}
        
        # Reverse map YF -> list of Internal (since multiple internal can map to same YF)
        # Actually usually 1-to-1 or N-to-1.
        # We want to display Internal Symbols.
        
        for internal_sym, yf_sym in yf_symbols_map.items():
            if yf_sym in hist_data:
                df_sym = hist_data[yf_sym]
                if not df_sym.empty and "price" in df_sym.columns:
                    # Rename series to internal symbol
                    series = df_sym["price"].copy()
                    # CRITICAL: Normalize index to DatetimeIndex to ensure alignment
                    series.index = pd.to_datetime(series.index)
                    price_series_dict[internal_sym] = series
        
        if not price_series_dict:
            return {"assets": [], "correlation": []}
            
        # Create aligned DataFrame
        combined_df = pd.DataFrame(price_series_dict)
        combined_df.sort_index(inplace=True) # Ensure sorted by date for pct_change
        
        # Log data stats for debugging
        logging.info(f"Correlation: Combined DF Shape: {combined_df.shape}")
        if not combined_df.empty:
             logging.info(f"Correlation: Date Range: {combined_df.index.min()} to {combined_df.index.max()}")
             logging.info(f"Correlation: Head:\n{combined_df.head()}")
        
        # Drop rows with any NaNs (strict correlation, valid comparison only)
        # Or pairwise? Pandas .corr() handles pairwise automatically handling NaNs.
        # But let's verify. Yes, default is excludes NaNs.
        
        # Calculate daily returns for correlation (Price correlation or Return correlation?)
        # Usually for finance, we use *Return Correlation*. Prices are non-stationary.
        returns_df = combined_df.pct_change().dropna()
        
        if returns_df.empty:
             return {"assets": [], "correlation": []}

        # 6. Calculate Correlation Matrix
        corr_matrix = returns_df.corr(method='pearson')
        
        # 7. Format for Frontend
        # Format: list of {x, y, value}
        
        assets = corr_matrix.columns.tolist() # Sorted list of symbols
        assets.sort() 
        
        # Re-sort matrix based on sorted assets for consistency
        corr_matrix = corr_matrix.reindex(index=assets, columns=assets)
        
        matrix_data = []
        for x in assets:
            for y in assets:
                val = corr_matrix.loc[x, y]
                # Handle NaN correlation (e.g. constant price)
                if pd.isna(val):
                    val = 0.0 # Or null
                
                matrix_data.append({
                    "x": x,
                    "y": y,
                    "value": round(float(val), 4)
                })
                
        return {
            "assets": assets,
            "correlation": matrix_data
        }

    except Exception as e:
        logging.error(f"Error calculating correlation matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/holdings")
async def get_holdings(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    show_closed: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the list of current holdings.

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names to filter by.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of holding records with calculated metrics.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _
    ) = data
    
    if df.empty:
        return []

    try:
        # Use helper but with show_closed_positions=False (though internal helper uses True, 
        # but calculate_portfolio_summary in logic handles it in summary_df_final)
        # Wait, the internal helper currently uses True. Let's fix that or pass it.
        
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=show_closed, # Pass the parameter
            data=data,
            current_user=current_user
        )
        
        summary_df = summary_data.get("summary_df")
        holdings_dict = summary_data.get("holdings_dict")
        
        if summary_df is None or summary_df.empty:
            return []

        # Filter closed positions if needed (the logic in portfolio_logic already does it for summary_df_final 
        # if show_closed_positions is False, but here we might want to be explicit or match behavior)
        # Actually calculate_portfolio_summary returns summary_df_final which is filtered.
        # But my helper currently uses show_closed_positions=True.
        
        # I'll update the helper to accept show_closed_positions.

        # Convert DataFrame to list of dicts
        # Handle NaNs
        summary_df = summary_df.where(pd.notnull(summary_df), None)
        
        # We need to make sure we return a clean list of dicts
        records = summary_df.to_dict(orient="records")
        
        # --- Merge 'lots' from holdings_dict into records ---
        if holdings_dict:
             for record in records:
                 sym = record.get("Symbol")
                 acct = record.get("Account")
                 if sym and acct:
                     key = (str(sym), str(acct))
                     # Try exact match first
                     if key in holdings_dict:
                         record["lots"] = holdings_dict[key].get("lots", [])
                     else:
                         # Fallback for case sensitivity or formatting issues
                         # This might be slow but safe
                        for h_key, h_data in holdings_dict.items():
                             if str(h_key[0]).lower() == str(sym).lower() and str(h_key[1]).lower() == str(acct).lower():
                                 record["lots"] = h_data.get("lots", [])
                                 break
        # ----------------------------------------------------

        return clean_nans(records)
        
    except Exception as e:
        logging.error(f"Error getting holdings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions")
async def get_transactions(
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the list of transactions, optionally filtered by account.

    Args:
        accounts (List[str], optional): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of transaction records.
    """
    df, _, _, _, _, _, _ = data
    
    if df.empty:
        return []

    try:
        # Filter by accounts if provided
        if accounts:
            df = df[df["Account"].isin(accounts)]

        # Sort by Date descending
        if "Date" in df.columns:
            df = df.sort_values(by="Date", ascending=False)

        # Handle NaNs and convert to list of dicts
        df = df.where(pd.notnull(df), None)
        
        # Ensure we include the ID in the response if it's in the index or a column
        if "original_index" in df.columns:
             # Make sure original_index is available as 'id' for the frontend
             df["id"] = df["original_index"]
        elif df.index.name == "original_index" or "original_index" in df.index.names:
             df["id"] = df.index.get_level_values("original_index")
        
        records = df.to_dict(orient="records")
        return clean_nans(records)
        
    except Exception as e:
        logging.error(f"Error getting transactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class TransactionInput(BaseModel):
    Date: str
    Type: str
    Symbol: str
    Quantity: float
    Price_Share: float = Field(0.0, alias="Price/Share")
    Total_Amount: Optional[float] = Field(None, alias="Total Amount")
    Commission: float = 0.0
    Account: str
    Split_Ratio: Optional[float] = Field(None, alias="Split Ratio")
    Note: Optional[str] = None
    Local_Currency: str = Field(..., alias="Local Currency")
    To_Account: Optional[str] = Field(None, alias="To Account")
    Tags: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True)

@router.post("/transactions")
async def create_transaction(
    transaction: TransactionInput,
    data: tuple = Depends(get_transaction_data)
):
    """
    Creates a new transaction.

    Args:
        transaction (TransactionInput): The transaction data payload.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message and the new transaction ID.
    """
    try:
        _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
            
        # Convert Pydantic model to dict with correct keys for DB
        tx_data = transaction.dict(by_alias=True)
        
        # Explicitly handle Price/Share alias if Pydantic didn't cover it fully or for safety
        if "Price_Share" in tx_data:
             tx_data["Price/Share"] = tx_data.pop("Price_Share")
        if "Total_Amount" in tx_data:
             tx_data["Total Amount"] = tx_data.pop("Total_Amount")
        if "Local_Currency" in tx_data:
             tx_data["Local Currency"] = tx_data.pop("Local_Currency")
        if "To_Account" in tx_data:
             tx_data["To Account"] = tx_data.pop("To_Account")
        if "Split_Ratio" in tx_data:
             tx_data["Split Ratio"] = tx_data.pop("Split_Ratio")
        if "Tags" in tx_data and tx_data["Tags"] is not None:
             # Ensure stripped string
             tx_data["Tags"] = str(tx_data["Tags"]).strip()

        success, new_id = add_transaction_to_db(conn, tx_data)
        conn.close()
        
        if success:
            reload_data_and_clear_cache() # Refresh transaction and summary caches
            return {"status": "success", "id": new_id, "message": "Transaction added"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add transaction to database")
            
    except Exception as e:
        logging.error(f"Error adding transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/transactions/{transaction_id}")
async def update_transaction(
    transaction_id: int,
    transaction: TransactionInput,
    data: tuple = Depends(get_transaction_data)
):
    """
    Updates an existing transaction.

    Args:
        transaction_id (int): The ID of the transaction to update.
        transaction (TransactionInput): The updated transaction data.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message.
    """
    try:
        _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
             raise HTTPException(status_code=500, detail="Database connection failed")

        tx_data = transaction.dict(by_alias=True)
        
        # Handle Aliases
        if "Price_Share" in tx_data:
             tx_data["Price/Share"] = tx_data.pop("Price_Share")
        if "Total_Amount" in tx_data:
             tx_data["Total Amount"] = tx_data.pop("Total_Amount")
        if "Local_Currency" in tx_data:
             tx_data["Local Currency"] = tx_data.pop("Local_Currency")
        if "To_Account" in tx_data:
             tx_data["To Account"] = tx_data.pop("To_Account")
        if "Split_Ratio" in tx_data:
             tx_data["Split Ratio"] = tx_data.pop("Split_Ratio")
        if "Tags" in tx_data and tx_data["Tags"] is not None:
             tx_data["Tags"] = str(tx_data["Tags"]).strip()

        success = update_transaction_in_db(conn, transaction_id, tx_data)
        conn.close()
        
        if success:
            reload_data_and_clear_cache()
            return {"status": "success", "message": "Transaction updated"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found or update failed")

    except Exception as e:
        logging.error(f"Error updating transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    data: tuple = Depends(get_transaction_data)
):
    """
    Deletes a transaction.

    Args:
        transaction_id (int): ID of the transaction to delete.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message.
    """
    try:
        _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
             raise HTTPException(status_code=500, detail="Database connection failed")

        success = delete_transaction_from_db(conn, transaction_id)
        conn.close()
        
        if success:
            reload_data_and_clear_cache()
            return {"status": "success", "message": "Transaction deleted"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found or delete failed")

    except Exception as e:
        logging.error(f"Error deleting transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class HoldingTagUpdate(BaseModel):
    account: str
    symbol: str
    tags: str

    model_config = ConfigDict(populate_by_name=True)

@router.post("/holdings/update_tags")
async def update_holding_tags(
    update_data: HoldingTagUpdate,
    data: tuple = Depends(get_transaction_data)
):
    """
    Updates tags for all transactions associated with a specific holding (Symbol + Account).
    """
    try:
        _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
             raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        # Clean tags
        tags_value = update_data.tags.strip()
        
        # Update all transactions for this symbol and account
        # Note: We probably want to update ALL types (Buy, Sell, Div, etc) so they stay grouped?
        # Or just open positions?
        # ShareSight groups by holding. So updating all history is consistent.
        sql = "UPDATE transactions SET Tags = ? WHERE Symbol = ? AND Account = ?"
        cursor.execute(sql, (tags_value, update_data.symbol, update_data.account))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        reload_data_and_clear_cache()
        return {"status": "success", "message": f"Updated tags for {rows_affected} transactions"}

    except Exception as e:
        logging.error(f"Error updating holding tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/ibkr")
async def sync_ibkr(
    current_user: User = Depends(get_current_user),
    data: tuple = Depends(get_transaction_data),
    config_manager = Depends(get_config_manager)
):
    """
    Syncs transactions from IBKR via Flex Web Service.
    """
    try:
        # Prioritize values from config_manager (persisted in manual_overrides.json)
        # Fall back to config.py (environment variables)
        token = config_manager.manual_overrides.get("ibkr_token") or config.IBKR_TOKEN
        query_id = config_manager.manual_overrides.get("ibkr_query_id") or config.IBKR_QUERY_ID
        
        if not token or not query_id:
            # We check here to provide a clear error to the user via the API
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "code": "CONFIG_MISSING",
                    "message": "IBKR API not configured. Please set IBKR Token and Query ID in your settings."
                }
            )
            
        _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
            
        connector = IBKRConnector(token=token, query_id=query_id)
        # Flex sync involves network I/O, run in threadpool to avoid blocking event loop
        try:
            new_transactions = await run_in_threadpool(connector.sync)
        except Exception as sync_err:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "code": "SYNC_FAILED",
                    "message": str(sync_err)
                }
            )
        
        if not new_transactions:
             return {
                 "status": "success", 
                 "message": "Sync successful, but no new transactions were found in the report.", 
                 "added_count": 0
             }
             
        staged_count = 0
        duplicate_count = 0
        
        # Helper logic for staging (since we had issues adding it to db_utils)
        def _stage_tx(conn, tx_data, u_id):
            ext_id = tx_data.get("ExternalID")
            cursor = conn.cursor()
            if ext_id:
                # Check main table
                cursor.execute("SELECT id FROM transactions WHERE ExternalID = ?", (ext_id,))
                if cursor.fetchone(): return False, "duplicate_main"
                # Check pending table
                cursor.execute("SELECT id FROM pending_transactions WHERE ExternalID = ?", (ext_id,))
                if cursor.fetchone(): return False, "duplicate_pending"
            
            db_cols = ["Date", "Type", "Symbol", "Quantity", "Price/Share", "Total Amount", "Commission", "Account", "Split Ratio", "Note", "Local Currency", "To Account", "Tags", "ExternalID", "user_id"]
            placeholders = ", ".join([f":{c.replace('/', '_').replace(' ', '_')}" for c in db_cols])
            cols_str = ", ".join([f'"{c}"' for c in db_cols])
            
            sql = f"INSERT INTO pending_transactions ({cols_str}) VALUES ({placeholders});"
            sql_data = {}
            for col in db_cols:
                val = tx_data.get(col) if col != "user_id" else u_id
                
                # Fallback for Total Amount if missing (or provided as 'Amount')
                if col == "Total Amount" and val is None:
                    val = tx_data.get("Amount") # Try alternate key
                    if val is None: # Calculate
                        q = tx_data.get("Quantity", 0)
                        p = tx_data.get("Price/Share", 0)
                        c = tx_data.get("Commission", 0)
                        t = tx_data.get("Type", "").upper()
                        if q and p:
                            val = (q * p) + (c if t == "BUY" else -c)

                if col == "Type" and isinstance(val, str): val = val.strip().title()
                if col == "Date" and isinstance(val, (datetime, date)): val = val.strftime("%Y-%m-%d")
                sql_data[col.replace('/', '_').replace(' ', '_')] = None if pd.isna(val) else val
            
            cursor.execute(sql, sql_data)
            return True, cursor.lastrowid

        for tx_data in new_transactions:
            success, result = _stage_tx(conn, tx_data, current_user.id)
            if success:
                staged_count += 1
            elif result in ["duplicate_main", "duplicate_pending"]:
                duplicate_count += 1
                    
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": f"Sync successful. {staged_count} transactions staged for review ({duplicate_count} skipped as duplicates).",
            "staged_count": staged_count,
            "duplicate_count": duplicate_count
        }
        
    except Exception as e:
        logging.error(f"Error during IBKR sync: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/sync/ibkr/pending")
async def get_pending_ibkr(
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Fetch pending transactions for review."""
    try:
        query = "SELECT * FROM pending_transactions WHERE user_id = ? ORDER BY Date DESC"
        df = pd.read_sql_query(query, conn, params=(current_user.id,))
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/ibkr/approve")
async def approve_ibkr(
    ids: List[int],
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Approve and move transactions from staging to main table."""
    try:
        cursor = conn.cursor()
        approved_count = 0
        
        # Columns in pending (exclude ID)
        cols = ["Date", "Type", "Symbol", "Quantity", "Price/Share", "Total Amount", "Commission", "Account", "Split Ratio", "Note", "Local Currency", "To Account", "Tags", "ExternalID", "user_id"]
        cols_str = ", ".join([f'"{c}"' for c in cols])
        
        for p_id in ids:
            # Fetch from pending
            cursor.execute(f"SELECT {cols_str} FROM pending_transactions WHERE id = ? AND user_id = ?", (p_id, current_user.id))
            row = cursor.fetchone()
            if row:
                # Insert into main table
                placeholders = ", ".join(["?"] * len(cols))
                cursor.execute(f"INSERT INTO transactions ({cols_str}) VALUES ({placeholders})", row)
                # Delete from pending
                cursor.execute("DELETE FROM pending_transactions WHERE id = ?", (p_id,))
                approved_count += 1
        
        conn.commit()
        if approved_count > 0:
            reload_data_and_clear_cache()
            
        return {"status": "success", "message": f"Successfully approved {approved_count} transactions.", "count": approved_count}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/ibkr/reject")
async def reject_ibkr(
    ids: List[int],
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Discard pending transactions."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM pending_transactions WHERE id IN ({','.join(['?']*len(ids))}) AND user_id = ?", (*ids, current_user.id))
        deleted_count = cursor.rowcount
        conn.commit()
        if deleted_count > 0:
            reload_data_and_clear_cache()
        return {"status": "success", "message": f"Discarded {deleted_count} transactions.", "count": deleted_count}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
        logging.error(f"Error during IBKR sync: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync error: {str(e)}")


def _calculate_historical_performance_internal(
    currency: str,
    period: str,
    accounts: Optional[List[str]],
    benchmarks: List[str],
    data: tuple,
    return_df: bool = False,
    interval: str = "1d",
    from_date_str: Optional[str] = None,
    to_date_str: Optional[str] = None
):
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _
    ) = data

    if df.empty:
        return pd.DataFrame() if return_df else []

    # Determine date range
    from_date_custom = None
    to_date_custom = None
    if from_date_str:
        try:
            from_date_custom = datetime.strptime(from_date_str, "%Y-%m-%d").date()
        except Exception:
            logging.warning(f"Invalid from_date_str: {from_date_str}")
    if to_date_str:
        try:
            to_date_custom = datetime.strptime(to_date_str, "%Y-%m-%d").date()
        except Exception:
            logging.warning(f"Invalid to_date_str: {to_date_str}")

    # If to_date (custom) is not provided:
    # 1. For "1d" view (Intraday): Use get_latest_trading_date(). 
    #    This returns "Yesterday" if we are in US pre-market (e.g. 7 AM EST), preventing a flat, empty "Today" graph.
    # 2. For other views (1m, 1y, etc): Use get_est_today().
    #    This ensures "Today" is included as the last data point, capturing international markets (SET) 
    #    that may have already closed/traded, matching the Summary Card total.
    from utils_time import get_est_today, get_latest_trading_date, is_tradable_day, get_nyse_calendar
    
    # UNIFIED: Use get_est_today() + 1 day as the exclusive end_date for all relative periods.
    # This ensures yfinance includes today's data (if available) and the graph isn't truncated.
    if to_date_custom:
        end_date = to_date_custom
    else:
        end_date = get_est_today() + timedelta(days=1)
    
    # Calculate start_date based on period (legacy logic still valid, relative to end_date)
    if period == "custom" and from_date_custom:
        start_date = from_date_custom
    elif period == "1d":
        # Handle weekends/Mondays to ensure we get some data (last trading day)
        # If Sat(5) or Sun(6), go back to Friday.
        # If Mon(0), go back to Friday to ensure we have context if market just opened.
        if end_date.weekday() == 6: # Sunday -> Friday
             start_date = end_date - timedelta(days=2)
             end_date = end_date + timedelta(days=1)
        elif end_date.weekday() == 5: # Saturday -> Friday
             start_date = end_date - timedelta(days=1)
             end_date = end_date + timedelta(days=1)
        # REMOVED: Monday context logic (start_date = end_date - 3)
        # Monday will now fall through to the 'else' block, treating it as a standard day
        # (Start = Monday, End = Tuesday), ensuring 1D graph shows ONLY Monday.
        if to_date_custom:
             start_date = end_date
        else:
             # Standard Weekday 1D View
             # end_date is already Today+1. We want start_date to be "Latest Trading Date".
             start_date = get_latest_trading_date()
             # Keep end_date as Today+1 (exclusive)
    elif period == "5d" or period == "7d":
        start_date = end_date - timedelta(days=7)
    elif period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "3y":
        start_date = end_date - timedelta(days=365 * 3)
    elif period == "5y":
        start_date = end_date - timedelta(days=365 * 5)
    elif period == "10y":
        start_date = end_date - timedelta(days=365 * 10)
    elif period == "ytd":
        start_date = date(end_date.year, 1, 1)
    elif period == "all":
        start_date = df["Date"].min().date()
    else:
        start_date = end_date - timedelta(days=365)

    # --- NEW: CLAMP START DATE TO ACCOUNT HISTORY ---
    # If specific accounts are selected, or even for "All" view, 
    # we want the graph to start where the data actually begins for those accounts,
    # rather than showing a long flat line from the global portfolio inception (2002).
    if not df.empty and "Date" in df.columns:
        earliest_dt = None
        earliest_dt = None
        if accounts and "Account" in df.columns:
             unique_accts = df["Account"].unique()
             logging.info(f"DEBUG_CLAMP: Received accounts={accounts}. Available in DF: {unique_accts}")
             
             # Find earliest date relevant to selected accounts
             mask = df["Account"].isin(accounts)
             if mask.any():
                 earliest_dt = df.loc[mask, "Date"].min().date()
                 logging.info(f"DEBUG_CLAMP: Match found. Earliest date: {earliest_dt}")
             else:
                 logging.info("DEBUG_CLAMP: No transactions found for selected accounts.")
        
        # Fallback: if no specific accounts or filtering failed, use global min
        
        # Fallback: if no specific accounts or filtering failed, use global min
        if earliest_dt is None and period == "all": 
             earliest_dt = df["Date"].min().date()

        # Apply clamp: If requested start_date (e.g. 5Y ago) is before 
        # the account actually existed, shift to inception.
        logging.info(f"DEBUG_CLAMP: Accounts={accounts}, Period={period}, RequestStart={start_date}, EarliestDT={earliest_dt}")
        if earliest_dt and start_date < earliest_dt:
             start_date = earliest_dt
             logging.info(f"DEBUG_CLAMP: Clamped start_date to {start_date}")
        else:
             logging.info("DEBUG_CLAMP: No clamping applied.")

    # --- NEW: BASELINE BUFFERING ($t_{-1}$) ---
    # Store the intended display start date before we buffer it for calculations
    display_start_date = start_date
    
    # To support "Change relative to previous point", we expand the requested start_date 
    # back by one interval. This ensures the backend has the baseline point available 
    # for normalization and the chart can render the performance during the first day.
    # --- CHANGE: Always calculate full history for Daily/Weekly/Monthly ---
    # To ensures the portfolio state (cash, cost basis) is correctly built up from the beginning,
    # we force the CALCULATION start date to be the earliest transaction date.
    # The graph will be sliced back to the 'display_start_date' for the UI at the end.
    is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
    
    if not is_intraday:
        if not df.empty and "Date" in df.columns:
             # Force calculation from inception
             start_date = df["Date"].min().date()
    elif period != "all":
        # For intraday, we still buffer by a small amount (e.g. 1 day) 
        # because calculating intraday from inception would be too heavy/impossible.
        start_date = start_date - timedelta(days=1)
            
    # --- END BASELINE BUFFERING ---

    # Call calculation logic
    # Revert to daily interval for ALL periods to prevent repeating dates in graph
    # --- MODIFIED: Use provided interval or default ---
    # Force "D" for "all" period to match Desktop App (main_gui.py) logic exactly
    # This ensures they share the exact same cache file (which is known to be clean).
    # --- MAPPING: Convert benchmark display names to YF tickers ---
    ticker_to_name = {}
    if benchmarks:
        mapped_benchmarks = []
        # Create a lowercase mapping for case-insensitive lookup
        bm_mapping_lower = {k.lower(): v for k, v in BENCHMARK_MAPPING.items()}
        yf_map_lower = {k.lower(): v for k, v in YFINANCE_INDEX_TICKER_MAP.items()}
        
        for b in benchmarks:
            b_lower = b.lower()
            # Check full names (from UI) first
            if b_lower in bm_mapping_lower:
                ticker = bm_mapping_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            # Check short codes (legacy)
            elif b_lower in yf_map_lower:
                ticker = yf_map_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            else:
                # If already a ticker or unknown, keep as is
                mapped_benchmarks.append(b)
        benchmarks = mapped_benchmarks

    logging.info(f"API History: Mapped benchmarks: {benchmarks}. TickerToName: {ticker_to_name}")

    logging.info(f"API History: Fetching period='{period}', interval='{interval}', start={start_date}, end={end_date}, benchmarks={benchmarks}")
    
    calc_interval = "D" if period == "all" else interval
    
    t_start = time.time()
    try:
        daily_df, _, historical_fx_yf, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=None, # Not needed for web view
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=start_date,
            end_date=end_date,
            display_currency=currency,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            include_accounts=accounts,
            benchmark_symbols_yf=benchmarks,
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            interval=calc_interval,
            original_csv_file_path=original_csv_path
        )
        logging.info(f"API History: Calc returned daily_df with {len(daily_df)} rows. Columns: {list(daily_df.columns)}")
        if not daily_df.empty:
            logging.info(f"API History: daily_df tail:\n{daily_df.tail(3).to_string()}")
    except Exception as e:
        logging.error(f"API Error in calculate_historical_performance: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    logging.info(f"API: calculation complete in {time.time() - t_start:.2f}s.")
    
    if daily_df is None or daily_df.empty:
        return pd.DataFrame() if return_df else []

    # --- NEW: BASELINE FILTERING ---
    if daily_df is not None and not daily_df.empty:
        # --- FIX: Ensure Datetime Index ---
        # calculate_historical_performance might return a reset index with 'date' or 'Date' column
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            logging.info(f"DEBUG_INDEX: Index is {type(daily_df.index)}. Columns: {daily_df.columns}")
            if "Date" in daily_df.columns:
                daily_df.set_index("Date", inplace=True)
                logging.info("DEBUG_INDEX: Set index to 'Date'")
            elif "date" in daily_df.columns:
                daily_df.set_index("date", inplace=True)
                daily_df.index.name = "Date" # Standardize name
                logging.info("DEBUG_INDEX: Set index to 'date'")
            else:
                logging.warning("DEBUG_INDEX: No Date/date column found to set index!")
        
        # Ensure UTC alignment for comparison and output
        daily_df.index = pd.to_datetime(daily_df.index, utc=True)
        
        # --- FIX: Handle filtered history (NaNs) ---
        # If calculating TWR for an account that didn't exist in 2002, values might be NaN. 
        # Fill with 1.0 (baseline wealth) so graph isn't broken.
        if "Portfolio Accumulated Gain" in daily_df.columns:
             daily_df["Portfolio Accumulated Gain"] = daily_df["Portfolio Accumulated Gain"].fillna(1.0)
        
        # Ensure we only have ONE baseline point before the display_start_date.
        # This prevents plotting "previous period data" while keeping the anchor.
        # Modified: Apply this to "all" as well so it respects the clamped display_start_date
        if True: # period != "all":
            try:
                # Use localized Timestamp for comparison
                ds_ts = pd.Timestamp(display_start_date, tz='UTC')
                
                df_before = daily_df[daily_df.index < ds_ts]
                df_display = daily_df[daily_df.index >= ds_ts]
                
                if not df_before.empty:
                    # Take only the LAST point before display_start_date as the baseline
                    baseline_point = df_before.iloc[[-1]]
                    daily_df = pd.concat([baseline_point, df_display])
                    logging.info(f"API: Filtered baseline to 1 point at {baseline_point.index[0]}")
                else:
                    daily_df = df_display
            except Exception as e_filter:
                logging.error(f"API: Baseline filtering failed: {e_filter}")
    # --- END BASELINE FILTERING ---

    # --- NEW: NORMALIZATION (Re-base to 0% at baseline) ---
    # With full history calculation, the values at the start of a 1Y period might be 1.5 (50% gain).
    # We must normalize so the graph starts at 1.0 (0% gain).
    # We use iloc[0] (the baseline point t-1) as the reference.
    if daily_df is not None and not daily_df.empty: # Modified: Always normalize, even for "all"
        try:
             # 1. Normalize Portfolio TWR
             if "Portfolio Accumulated Gain" in daily_df.columns:
                 start_val = daily_df["Portfolio Accumulated Gain"].iloc[0]
                 if start_val != 0:
                     daily_df["Portfolio Accumulated Gain"] = daily_df["Portfolio Accumulated Gain"] / start_val
                 # Optimization: No need to log every time, but good for debugging if needed
                 # logging.debug(f"API: Normalized Portfolio TWR by factor {start_val}")

             # 2. Normalize Benchmarks
             # 2. Normalize Benchmarks
             if benchmarks:
                 for b_ticker in benchmarks:
                     bm_col = f"{b_ticker} Accumulated Gain"
                     if bm_col in daily_df.columns:
                         # Backfill to ensure we have a valid start value if the exact start date is missing data
                         daily_df[bm_col] = daily_df[bm_col].bfill()
                         
                         bm_start_val = daily_df[bm_col].iloc[0]
                         if pd.notna(bm_start_val) and bm_start_val != 0:
                             daily_df[bm_col] = daily_df[bm_col] / bm_start_val
                         elif pd.isna(bm_start_val):
                              # If still NaN (empty series?), fill with 1.0
                              daily_df[bm_col] = 1.0
                     # Add logging for benchmark price series alignment and Tuesday data
                     bm_price_col = f"{b_ticker} Price"
                     if bm_price_col in daily_df.columns:
                         num_tuesday = daily_df.loc[daily_df.index.date == date(2026, 1, 20), bm_price_col].notna().sum()
                         logging.info(f"Benchmark {b_ticker}: Aligned. Tuesday points: {num_tuesday}")

        except Exception as e_norm:
            logging.error(f"API: Normalization failed: {e_norm}")
    # --- END NORMALIZATION ---

    if return_df:
        # Standardize for internal consumers (like portfolio health)
        df_ret = daily_df.copy()
        if "Portfolio Value" in df_ret.columns:
            df_ret.rename(columns={"Portfolio Value": "value"}, inplace=True)
        df_ret.index.name = "date"
        return df_ret.reset_index()

    # Format Result for Recharts
    # We need "date" (str), "value" (float), "twr" (float), and benchmarks
    
    drawdown_series = calculate_drawdown_series(daily_df["Portfolio Value"])
    
    result = []
    # daily_df index is Date

    ticker_to_name = {v: k for k, v in config.BENCHMARK_MAPPING.items()}
    
    # --- FX Rate series preparation ---
    fx_rate_series = None
    if currency and currency.upper() != "USD" and historical_fx_yf:
        fx_pair = f"{currency.upper()}=X"
        if fx_pair in historical_fx_yf:
            fx_df = historical_fx_yf[fx_pair]
            # Handle both 'price' (new format) and 'Close' (old/direct format)
            rate_col = "price" if "price" in fx_df.columns else ("Close" if "Close" in fx_df.columns else ("Adj Close" if "Adj Close" in fx_df.columns else None))
            
            if not fx_df.empty and rate_col:
                # Ensure UTC alignment
                fx_idx = fx_df.index
                if not isinstance(fx_idx, pd.DatetimeIndex) or fx_idx.tz is None:
                    fx_idx = pd.to_datetime(fx_idx, utc=True)
                
                fx_df_proc = fx_df.copy()
                fx_df_proc.index = fx_idx
                
                # Reindex with forward fill to match portfolio timestamps
                fx_rate_series = fx_df_proc[rate_col].reindex(daily_df.index, method='ffill')
    
    
    # Debug logging for API filtering
    filtered_count = 0
    pre_market_count = 0
    post_market_count = 0
    total_input = len(daily_df)
    
    # ds_ts is used to identify baseline points
    ds_ts = pd.Timestamp(display_start_date, tz='UTC')
    
    # Pre-calculate tradable days for the range to optimize the loop
    # (Avoids calling cal.schedule thousands of times for intraday data)
    try:
        cal = get_nyse_calendar()
        tradable_days = set(cal.schedule(start_date=start_date, end_date=end_date).index.date)
    except Exception as e_cal:
        logging.warning(f"API: Failed to pre-calculate tradable days: {e_cal}")
        tradable_days = None

    for i, (dt, row) in enumerate(daily_df.iterrows()):
        # Ensure the FIRST point (the baseline t-1) is correctly identified.
        # It's only a baseline if it's before the intended display range.
        is_baseline = (i == 0 and period != "all" and dt < ds_ts)
        
        if not is_baseline:
            # Skip weekends and market holidays to avoid flat lines in graph
            if tradable_days is not None:
                if dt.date() not in tradable_days:
                    continue
            elif not is_tradable_day(dt):
                continue
        # Strict market hours filter for intraday (09:30 - 16:00 EST)
        # We only apply this to intraday intervals.
        # EXCEPTION: Always allow the baseline point (i=0) to provide the 0% anchor.
        if not is_baseline and interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
             # Convert dt to NY time for filtering
             # dt in daily_df.index is already UTC-aware Timestamp
             dt_ny = dt.tz_convert("America/New_York")
             if not (dt_time(9, 30) <= dt_ny.time() <= dt_time(16, 0)):
                 if dt_ny.time() < dt_time(9, 30): 
                     if pre_market_count == 0: logging.info(f"First Pre-Market Reject: {dt_ny} (from {dt})")
                     pre_market_count += 1
                 else: 
                     post_market_count += 1
                 continue
             else:
                 if filtered_count == 0: logging.info(f"First Accepted: {dt_ny} (from {dt})")

        # Handle NaN values
        val = row.get("Portfolio Value", 0.0)
        twr = row.get("Portfolio Accumulated Gain", 1.0)
        # Use simple 'drawdown' column if available (new logic), else fallback or use series
        # Note: 'drawdown' from portfolio_logic is already in percentage (e.g. -5.0)
        dd = row.get("drawdown", 0.0) 
        
        # --- NEW: Money-weighted metrics ---
        abs_gain = row.get("Absolute Gain ($)", 0.0)
        abs_roi = row.get("Absolute ROI (%)", 0.0)
        cum_flow = row.get("Cumulative Net Flow", 0.0)

        item = {
            "date": dt.isoformat(),
            "is_baseline": is_baseline,
            "value": val if pd.notnull(val) else 0.0,
            "twr": (twr - 1) * 100 if pd.notnull(twr) else 0.0, # Convert to percentage change
            "drawdown": dd if pd.notnull(dd) else 0.0, # Already percentage from portfolio_logic
            "abs_gain": float(abs_gain) if pd.notnull(abs_gain) else 0.0,
            "abs_roi": float(abs_roi) if pd.notnull(abs_roi) else 0.0,
            "cum_flow": float(cum_flow) if pd.notnull(cum_flow) else 0.0,
        }

        # Add FX Rate if available
        if fx_rate_series is not None:
            fx_val = fx_rate_series.get(dt)
            if pd.notnull(fx_val):
                item["fx_rate"] = float(fx_val)
    

        # Add benchmark data
        if benchmarks:
            for b_ticker in benchmarks:
                bm_col = f"{b_ticker} Accumulated Gain"
                if bm_col in daily_df.columns:
                    b_val = row.get(bm_col)
                    display_name = ticker_to_name.get(b_ticker, b_ticker)
                    item[display_name] = (b_val - 1) * 100 if pd.notnull(b_val) else 0.0
            
        result.append(item)
        
    logging.info(f"API History: Input {total_input} rows. Filtered: Pre={pre_market_count}, Post={post_market_count}. Output {len(result)} rows.")
    return clean_nans(result)


@router.get("/history")
def get_history(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    period: str = "1y",
    benchmarks: Optional[List[str]] = Query(None),
    interval: str = "1d",
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns historical portfolio performance (Value and TWR) and benchmarks.
    """
    logging.info(f"get_history: period={period}, interval={interval}, from={from_date}, to={to_date}")
    try:
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)

        return _calculate_historical_performance_internal(
            currency=currency,
            period=period,
            accounts=accounts,
            benchmarks=mapped_benchmarks,
            data=data,
            return_df=False,
            interval=interval,
            from_date_str=from_date,
            to_date_str=to_date
        )
    except Exception as e:
        logging.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock_history/{symbol}")
async def get_stock_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns historical price data for a single stock, with optional benchmarks.
    """
    try:
        _, _, user_symbol_map, user_excluded_symbols, _, _, _ = data
        mdp = get_mdp()
        
        # 1. Map Symbol
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
        if not yf_symbol:
            # Try as direct ticker if not found in map (e.g. for benchmarks or unheld stocks)
            yf_symbol = symbol 

        # 2. Map Benchmarks
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)
        
        # 3. Determine Date Range
        # Using helper logic similar to _calculate_historical_performance_internal
        from utils_time import get_est_today, get_latest_trading_date
        
        # End Date: Today + 1 (exclusive) to ensure we get today's data
        end_date = get_est_today() + timedelta(days=1)
        
        # Start Date
        if period == "1d":
            interval = "2m" # Force Intraday
            # For 1D, we want the last trading session.
            # If today is trading, get today. If weekend, get Friday.
            latest_trading = get_latest_trading_date()
            start_date = latest_trading
            # For intraday '1d', end_date logic in market_data handles the "up to now"
            # But let's be explicit: start at latest_trading, end at latest_trading + 1
            end_date = latest_trading + timedelta(days=1)
        elif period == "5d":
            start_date = end_date - timedelta(days=7) # Go back a week to cover 5 trading days
            interval = "15m" # Higher res for 5d
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
            interval = "1d" # Daily is fine, or 60m? Daily is standard for 1M.
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "3y":
            start_date = end_date - timedelta(days=365*3)
        elif period == "5y":
            start_date = end_date - timedelta(days=365*5)
        elif period == "10y":
            start_date = end_date - timedelta(days=365*10)
        elif period == "ytd":
            # start_date = date(end_date.year, 1, 1) 
            # Better YTD: Start of current year
            today = get_est_today()
            start_date = date(today.year, 1, 1)
        elif period == "max" or period == "all":
             # Arbitrary long history
             start_date = date(1980, 1, 1)
        else:
            # Default to 1y if unknown
            start_date = end_date - timedelta(days=365)

        # 4. Fetch Data (Main Symbol + Benchmarks)
        symbols_to_fetch = [yf_symbol] + mapped_benchmarks
        
        # Use get_historical_data (handles DB sync and cache)
        # Note: For intraday (1m, 5m etc), get_historical_data bypasses DB write usually/reads from special table
        # We need to make sure interval is passed correct.
        
        hist_data, _ = mdp.get_historical_data(
            symbols_to_fetch,
            start_date,
            end_date,
            interval=interval
        )
        
        if yf_symbol not in hist_data or hist_data[yf_symbol].empty:
            # Fallback: Maybe it's a crypto or something that failed mapping?
            # Or just no data.
            return []

        # 5. Align and Process
        # We want a single list of dicts: { date, price, volume, bm1, bm2... }
        # Merge on index (Date)
        
        main_df = hist_data[yf_symbol].copy()
        main_df.rename(columns={"price": "value", "Volume": "volume"}, inplace=True)
        
        if "value" not in main_df.columns:
            # Fallback if rename failed or price missing
            if "price" in main_df.columns:
                main_df["value"] = main_df["price"]
            else:
                # Last resort: use first column
                if not main_df.empty:
                    main_df["value"] = main_df.iloc[:, 0]
                else:
                    return [] # Should be caught above

        if "volume" not in main_df.columns:
            main_df["volume"] = 0.0

        # --- NEW: Filter Intraday Data (Market Hours Only: 09:30 - 16:00 EST) ---
        if period in ["1d", "5d"] and not main_df.empty:
            try:
                # Ensure index is timezone-aware. yfinance usually returns tz-aware (America/New_York) for intraday.
                if main_df.index.tz is None:
                    main_df.index = main_df.index.tz_localize("America/New_York", ambiguous='infer')
                else:
                    main_df.index = main_df.index.tz_convert("America/New_York")
                
                # Filter strictly between 09:30 and 16:00
                main_df = main_df.between_time("09:30", "16:00")
            except Exception as e:
                logging.warning(f"Error filtering market hours for {symbol}: {e}")

        # Calculate Return % (Normalized to start)
        if not main_df.empty and "value" in main_df.columns:
            first_val = main_df["value"].iloc[0]
            if first_val and first_val > 0:
                main_df["return_pct"] = (main_df["value"] / first_val - 1) * 100
            else:
                main_df["return_pct"] = 0.0
                
        # Join Benchmarks
        cols_to_keep = ["value", "volume", "return_pct"]
        result_df = main_df[cols_to_keep].copy()
        
        for bm in mapped_benchmarks:
            if bm in hist_data and not hist_data[bm].empty:
                bm_df = hist_data[bm].copy()
                # Normalize benchmark
                if "price" in bm_df.columns:
                    bm_start = bm_df["price"].iloc[0]
                    if bm_start and bm_start > 0:
                         bm_series = (bm_df["price"] / bm_start - 1) * 100
                    else:
                         bm_series = 0.0
                         
                    # Reindex to match main_df (ffill for missing days if mismatched trading cals)
                    aligned_bm = bm_series.reindex(result_df.index, method='ffill')
                    result_df[bm] = aligned_bm

        # 6. Format for JSON
        # Reset index to get Date/timestamp column
        result_df = result_df.reset_index()
        # Rename index col to 'date' usually
        date_col = "date" if "date" in result_df.columns else ("Date" if "Date" in result_df.columns else "index")
        result_df.rename(columns={date_col: "date"}, inplace=True)
        
        # Convert date to isoformat
        result_df["date"] = result_df["date"].apply(lambda x: x.isoformat())
        
        records = result_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error serving stock history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capital_gains")
async def get_capital_gains(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the realized capital gains history.

    Args:
        currency (str): The display currency.
        accounts (List[str]): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of realized gain/loss records.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _
    ) = data

    if df.empty:
        return []

    try:
        # We need historical FX rates to calculate gains in display currency.
        # This requires running calculate_historical_performance.
        # We can optimize by asking for a shorter date range if needed, 
        # but for full history we need full range.
        
        # Determine full date range from transactions
        min_date = df["Date"].min().date()
        max_date = date.today()
        
        _, _, historical_fx_yf, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=min_date,
            end_date=max_date,
            interval="D",
            benchmark_symbols_yf=[], # No benchmarks needed
            display_currency=currency,
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            include_accounts=accounts,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            original_csv_file_path=original_csv_path
        )
        
        # Parse dates
        start_dt = None
        end_dt = None
        if from_date:
            try:
                start_dt = date.fromisoformat(from_date)
            except ValueError:
                pass
        if to_date:
            try:
                end_dt = date.fromisoformat(to_date)
            except ValueError:
                pass

        capital_gains_df = extract_realized_capital_gains_history(
            all_transactions_df=df,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            shortable_symbols=config.SHORTABLE_SYMBOLS,
            include_accounts=accounts,
            from_date=start_dt,
            to_date=end_dt
        )
        
        if capital_gains_df.empty:
            return []
            
        # Convert to list of dicts and clean NaNs
        # Ensure Date is string
        if 'Date' in capital_gains_df.columns:
            capital_gains_df['Date'] = capital_gains_df['Date'].astype(str)
            
        records = capital_gains_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting capital gains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dividends")
async def get_dividends(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the dividend history.

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of dividend records.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path,
        _
    ) = data

    if df.empty:
        return []

    try:
        # We need historical FX rates to calculate dividends in display currency.
        # This requires running calculate_historical_performance.
        
        # Determine full date range from transactions
        min_date = df["Date"].min().date()
        max_date = date.today()
        
        _, _, historical_fx_yf, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=min_date,
            end_date=max_date,
            interval="D",
            benchmark_symbols_yf=[], # No benchmarks needed
            display_currency=currency,
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            include_accounts=accounts,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            original_csv_file_path=original_csv_path
        )
        
        dividend_df = extract_dividend_history(
            all_transactions_df=df,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            include_accounts=accounts
        )
        
        if dividend_df.empty:
            return []
            
        # Convert to list of dicts and clean NaNs
        # Ensure Date is string
        if 'Date' in dividend_df.columns:
            dividend_df['Date'] = dividend_df['Date'].astype(str)
            
        records = dividend_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting dividends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/projected_income")
async def get_projected_income(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns projected monthly dividend income for the next 12 months.
    """
    try:
        # 1. Get Current Holdings
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=False,
            data=data,
            current_user=current_user
        )
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
            return []
            
        # 2. Extract Holdings & Quantities
        holdings = defaultdict(float) 
        rows = summary_df.to_dict(orient="records")
        for row in rows:
            sym = row.get("Symbol")
            if sym == "Total" or row.get("is_total"):
                continue
            qty = row.get("Quantity", 0)
            if qty > 0:
                holdings[sym] += qty
                
        if not holdings:
            return []

        # 3. Map to YF Symbols and Fetch Fundamentals
        df, _, user_symbol_map, user_excluded_symbols, _, _, _ = data
        from finutils import is_cash_symbol
        from market_data import MarketDataProvider, map_to_yf_symbol
        
        yf_symbols = set()
        sym_map = {} # Internal -> YF
        
        for sym in holdings.keys():
            if is_cash_symbol(sym):
                continue
            yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols)
            if yf_sym:
                yf_symbols.add(yf_sym)
                sym_map[sym] = yf_sym
                
        provider = get_mdp()
        fundamentals = provider.get_fundamental_data_batch(yf_symbols)
        
        # 4. Project Income
        # Structure: "YYYY-MM" -> {"total": float, "breakdown": {symbol: float}}
        projection = defaultdict(lambda: {"total": 0.0, "breakdown": defaultdict(float)})
        
        today = date.today()
        end_projection = today + pd.DateOffset(months=12)
        
        from datetime import timedelta
        
        for sym, qty in holdings.items():
            yf_sym = sym_map.get(sym)
            if not yf_sym or yf_sym not in fundamentals:
                continue
                
            fund = fundamentals[yf_sym]
            div_rate = fund.get("trailingAnnualDividendRate", 0.0)
            last_div_val = fund.get("lastDividendValue", 0.0)
            
            ex_div_ts = fund.get("exDividendDate")
            last_div_ts = fund.get("lastDividendDate")
            
            if not div_rate or div_rate <= 0:
                continue
                
            # Infer Frequency
            freq = 4 # Default to Quarterly
            amount_per_payment = 0.0
            
            if last_div_val > 0:
                ratio = div_rate / last_div_val
                if 0.5 <= ratio <= 1.5: freq = 1 # Annual
                elif 1.5 < ratio <= 2.5: freq = 2 # Semi
                elif 3.5 <= ratio <= 4.5: freq = 4 # Quarterly
                elif 11.0 <= ratio <= 13.0: freq = 12 # Monthly
                amount_per_payment = last_div_val
            else:
                amount_per_payment = div_rate / 4.0 # Fallback
                
            # Determine Starting Point
            start_dt = None
            if ex_div_ts and isinstance(ex_div_ts, (int, float)):
                 start_dt = date.fromtimestamp(ex_div_ts)
            elif last_div_ts and isinstance(last_div_ts, (int, float)):
                 start_dt = date.fromtimestamp(last_div_ts)
            
            if not start_dt:
                start_dt = today + timedelta(days=30)
            
            interval_days = 365 / freq
            current_dt = start_dt
            
            while current_dt < today:
                current_dt += timedelta(days=interval_days)
            
            while current_dt < end_projection.date():
                key = current_dt.strftime("%Y-%m")
                total_payment = amount_per_payment * qty
                
                projection[key]["total"] += total_payment
                projection[key]["breakdown"][sym] += total_payment
                
                current_dt += timedelta(days=interval_days)

        # 5. Format Result
        results = []
        iter_date = today.replace(day=1)
        for _ in range(12):
            key = iter_date.strftime("%Y-%m")
            label = iter_date.strftime("%b %Y")
            data_point = projection.get(key, {"total": 0.0, "breakdown": {}})
            
            # --- NEW: Add Cash Interest to Projection ---
            # We must fetch recent data to apply the same logic as dividend calendar
            if not data_point.get("breakdown"):
                data_point["breakdown"] = {}

            results.append({
                "month": label,
                "year_month": key,
                "value": data_point["total"],
                **data_point["breakdown"]
            })
            iter_date = (pd.Timestamp(iter_date) + pd.DateOffset(months=1)).date()
            
        # --- NEW: Inject Cash Interest into Results ---
        try:
            # Re-fetch settings similarly to get_dividend_calendar to ensure freshness
            # (data[1] might be stale if dependency cache invalidation lags)
            raw_df = data[0]
            
            config_manager = get_config_manager(current_user)
            config_manager.load_manual_overrides()
            interest_rates = config_manager.manual_overrides.get("account_interest_rates", {})
            thresholds = config_manager.manual_overrides.get("interest_free_thresholds", {})

            if interest_rates:
                
                cash_events = generate_cash_interest_events(
                    df=raw_df,
                    interest_rates=interest_rates,
                    thresholds=thresholds,
                    start_date=today,
                    end_date=end_projection.date()
                )
                
                # Aggregate events into the projection buckets
                for event in cash_events:
                    # event["dividend_date"] is 'YYYY-MM-DD'
                    ev_date = datetime.strptime(event["dividend_date"], "%Y-%m-%d").date()
                    key = ev_date.strftime("%Y-%m")
                    amt = event["amount"]
                    sym = event["symbol"] # $CASH
                    
                    # Find matching result bucket
                    for res in results:
                        if res["year_month"] == key:
                            current_val = res.get(sym, 0.0)
                            res[sym] = current_val + amt
                            res["value"] += amt # Update total
                            break
                            
        except Exception as e_cash:
             logging.error(f"Error adding cash interest to projection: {e_cash}")

        return clean_nans(results)

    except Exception as e:
        logging.error(f"Error projecting income: {e}", exc_info=True)
        return []

@router.get("/stock-analysis/{symbol}")
async def get_stock_analysis(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """
    Returns AI-powered stock analysis for a given symbol.
    """
    try:
        (_, _, user_symbol_map, user_excluded_symbols, _, _, _) = data
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols) or symbol
        
        mdp = get_mdp()
        # 1. Fetch Fundamentals
        fund_data = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        if not fund_data:
            fund_data = {}
            
        # 2. Fetch Release Statements and calculate ratios
        financials_df = mdp.get_financials(yf_symbol, "annual", force_refresh=force)
        balance_sheet_df = mdp.get_balance_sheet(yf_symbol, "annual", force_refresh=force)
        cashflow_df = mdp.get_cashflow(yf_symbol, "annual", force_refresh=force)
        
        ratios = {}
        if financials_df is not None and not financials_df.empty and balance_sheet_df is not None and not balance_sheet_df.empty:
            try:
                ratios_df = calculate_key_ratios_timeseries(
                    financials_df,
                    balance_sheet_df
                )
                if not ratios_df.empty:
                    # Take the most recent period ratios
                    ratios = ratios_df.iloc[0].to_dict()
            except Exception as e_ratio:
                logging.warning(f"Ratio calculation failed for analysis: {e_ratio}")

        # 3. Generate AI Review
        analysis = generate_stock_review(symbol, fund_data, ratios, force_refresh=force)
        
        # 4. Interactive Calculation of Intrinsic Value & Cache Update
        try:
            # Check cache first if not forced
            iv_results = None
            if not force and db_conn:
                try:
                    cached_results = get_cached_screener_results(db_conn, [symbol])
                    if symbol in cached_results:
                        cached_entry = cached_results[symbol]
                        
                        # Extract cached metadata
                        cached_fy_end = cached_entry.get("last_fiscal_year_end")
                        cached_mrq = cached_entry.get("most_recent_quarter")
                        cached_val_details_str = cached_entry.get("valuation_details")
                        
                        current_fy_end = fund_data.get("lastFiscalYearEnd")
                        current_mrq = fund_data.get("mostRecentQuarter")
                        
                        # Validate if cache is fresh enough (Timestamps match)
                        is_fresh = True
                        if current_fy_end and cached_fy_end != current_fy_end:
                            is_fresh = False
                        if current_mrq and cached_mrq != current_mrq:
                            is_fresh = False
                            
                        # Also ensure we actually have the detailed JSON stored
                        if is_fresh and cached_val_details_str:
                             logging.info(f"Using cached Intrinsic Value for {symbol} (Freshness verified)")
                             try:
                                 iv_results = json.loads(cached_val_details_str)
                                 # Ensure top-level metrics match cache (consistency check)
                                 # (optional, but good for safety)
                             except json.JSONDecodeError:
                                 logging.warning(f"Failed to decode cached valuation details for {symbol}")
                                 iv_results = None
                except Exception as e_cache_read:
                    logging.warning(f"Error checking IV cache for {symbol}: {e_cache_read}")

            if iv_results is None:
                # We calculate this ON THE FLY to ensure the frontend gets the detailed value
                # immediately when "Analyze" is clicked, updating the screener row live.
                logging.info(f"Recalculating Intrinsic Value for {symbol}...")
                iv_results = get_comprehensive_intrinsic_value(
                    fund_data, financials_df, balance_sheet_df, cashflow_df
                )
                
                # Serialize the entire result for storage
                iv_json = json.dumps(iv_results, default=str)
                
                # Inject into info dict so it gets picked up by update_intrinsic_value_in_cache
                # We copy it to avoid mutating the original fund_data for other uses if any
                info_for_update = fund_data.copy()
                info_for_update["valuation_details"] = iv_json

                # Update cache so screener table reads it next time (or live update listens to it)
                # db_conn is injected
                if db_conn:
                    update_intrinsic_value_in_cache(
                        db_conn,
                        symbol,
                        iv_results.get("average_intrinsic_value"),
                        iv_results.get("margin_of_safety_pct"),
                        fund_data.get("lastFiscalYearEnd"),
                        fund_data.get("mostRecentQuarter"),
                        info=info_for_update
                    )
            
            # Inject into response so frontend event can carry it
            analysis["intrinsic_value_data"] = iv_results

        except Exception as iv_e:
            logging.error(f"Failed to calculate IV during stock analysis for {symbol}: {iv_e}")

        return clean_nans(analysis)
    except Exception as e:
        logging.error(f"Error in stock analysis for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings")
async def get_settings(
    data: tuple = Depends(get_transaction_data),
    config_manager = Depends(get_config_manager)
):
    """
    Returns the current application configuration settings.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        _,
        _
    ) = data

    try:
        # Build a mapping of Symbol -> Local Currency from transactions
        symbol_to_currency = {}
        if not df.empty:
            # Get unique symbol/currency pairs from transactions
            symbol_curr_df = df[['Symbol', 'Local Currency']].drop_duplicates()
            symbol_to_currency = dict(zip(symbol_curr_df['Symbol'], symbol_curr_df['Local Currency']))

        # Enrich manual_overrides with currency info
        enriched_overrides = {}
        for symbol, override_data in manual_overrides.items():
            symbol_upper = symbol.upper()
            currency = symbol_to_currency.get(symbol_upper)
            
            # If not found in transactions, try to derive from symbol suffix or use default
            if not currency:
                if symbol_upper.endswith(".BK") or ":BKK" in symbol_upper:
                    currency = "THB"
                else:
                    currency = "USD" # Default to USD
            
            # Create a copy and add currency
            enriched_data = override_data.copy() if isinstance(override_data, dict) else {"price": override_data}
            enriched_data["currency"] = currency
            enriched_overrides[symbol_upper] = enriched_data

        groups = config_manager.gui_config.get("account_groups", {})

        return {
            "manual_overrides": enriched_overrides,
            "user_symbol_map": user_symbol_map,
            "user_excluded_symbols": list(user_excluded_symbols),
            "account_currency_map": account_currency_map,
            "account_groups": config_manager.gui_config.get("account_groups", {}),
            "available_currencies": config_manager.gui_config.get("available_currencies", ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY']),
            "account_interest_rates": config_manager.manual_overrides.get("account_interest_rates", {}),
            "interest_free_thresholds": config_manager.manual_overrides.get("interest_free_thresholds", {}),
            "valuation_overrides": config_manager.manual_overrides.get("valuation_overrides", {}),
            "visible_items": config_manager.gui_config.get("visible_items", []),
            "benchmarks": config_manager.gui_config.get("benchmarks", ['S&P 500', 'Dow Jones', 'NASDAQ']),
            "show_closed": config_manager.gui_config.get("show_closed", False),
            "display_currency": config_manager.gui_config.get("display_currency", "USD"),
            "selected_accounts": config_manager.gui_config.get("selected_accounts", []),
            "active_tab": config_manager.gui_config.get("active_tab", "performance"),
            "ibkr_token": config_manager.manual_overrides.get("ibkr_token") or config.IBKR_TOKEN,
            "ibkr_query_id": config_manager.manual_overrides.get("ibkr_query_id") or config.IBKR_QUERY_ID
        }
    except Exception as e:
        logging.error(f"Error getting settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class SettingsUpdate(BaseModel):
    manual_price_overrides: Optional[Dict[str, Any]] = None
    user_symbol_map: Optional[Dict[str, str]] = None
    user_excluded_symbols: Optional[List[str]] = None
    account_groups: Optional[Dict[str, List[str]]] = None
    account_currency_map: Optional[Dict[str, str]] = None
    available_currencies: Optional[List[str]] = None
    account_interest_rates: Optional[Dict[str, float]] = None
    interest_free_thresholds: Optional[Dict[str, float]] = None
    valuation_overrides: Optional[Dict[str, Any]] = None
    visible_items: Optional[List[str]] = None
    benchmarks: Optional[List[str]] = None
    show_closed: Optional[bool] = None
    display_currency: Optional[str] = None
    selected_accounts: Optional[List[str]] = None
    active_tab: Optional[str] = None
    ibkr_token: Optional[str] = None
    ibkr_query_id: Optional[str] = None

@router.post("/settings/update")
async def update_settings(
    settings: SettingsUpdate,
    config_manager = Depends(get_config_manager)
):
    """
    Updates the application settings (manual overrides, symbol map, exclude list).
    """
    try:
        # RELOAD DATA FROM DISK to ensure we have the latest state before merging updates
        config_manager.load_manual_overrides()
        
        current_overrides = config_manager.manual_overrides
        
        # Update Manual Price Overrides
        if settings.manual_price_overrides is not None:
             current_overrides["manual_price_overrides"] = settings.manual_price_overrides

        if settings.user_symbol_map is not None:
            current_overrides["user_symbol_map"] = settings.user_symbol_map
            
        if settings.user_excluded_symbols is not None:
            current_overrides["user_excluded_symbols"] = sorted(list(set(settings.user_excluded_symbols)))
            
        if settings.account_interest_rates is not None:
            current_overrides["account_interest_rates"] = settings.account_interest_rates
            
        if settings.interest_free_thresholds is not None:
            current_overrides["interest_free_thresholds"] = settings.interest_free_thresholds

        if settings.valuation_overrides is not None:
            current_overrides["valuation_overrides"] = settings.valuation_overrides

        if settings.ibkr_token is not None:
            current_overrides["ibkr_token"] = settings.ibkr_token
            
        if settings.ibkr_query_id is not None:
            current_overrides["ibkr_query_id"] = settings.ibkr_query_id

        # Update GUI Config (Dashboard persistence)
        gui_config_changed = False
        if settings.visible_items is not None:
            config_manager.gui_config["visible_items"] = settings.visible_items
            gui_config_changed = True
        
        if settings.benchmarks is not None:
            config_manager.gui_config["benchmarks"] = settings.benchmarks
            gui_config_changed = True
            
        if settings.show_closed is not None:
            config_manager.gui_config["show_closed"] = settings.show_closed
            gui_config_changed = True

        if settings.account_groups is not None:
            config_manager.gui_config["account_groups"] = settings.account_groups
            gui_config_changed = True
            
        if settings.account_currency_map is not None:
             config_manager.gui_config["account_currency_map"] = settings.account_currency_map
             gui_config_changed = True

        if settings.available_currencies is not None:
             config_manager.gui_config["available_currencies"] = settings.available_currencies
             gui_config_changed = True

        if settings.display_currency is not None:
             config_manager.gui_config["display_currency"] = settings.display_currency
             gui_config_changed = True

        if settings.selected_accounts is not None:
             config_manager.gui_config["selected_accounts"] = settings.selected_accounts
             gui_config_changed = True

        if settings.active_tab is not None:
             config_manager.gui_config["active_tab"] = settings.active_tab
             gui_config_changed = True

        if gui_config_changed:
            config_manager.save_gui_config()

        # Save to AppData
        if config_manager.save_manual_overrides(current_overrides):
            
            # --- Added: Mirror save to Project Root if file exists ---
            # This ensures the user's "source of truth" in their workspace stays in sync
            project_overrides_file = os.path.join(project_root, "manual_overrides.json")
            if os.path.exists(project_overrides_file):
                try:
                    with open(project_overrides_file, "w", encoding="utf-8") as f:
                        json.dump(current_overrides, f, indent=4, ensure_ascii=False)
                    logging.info(f"Mirrored settings update to {project_overrides_file}")
                except Exception as e:
                    logging.warning(f"Failed to mirror settings to project file: {e}")
            # ---------------------------------------------------------

            # Reload data to apply changes (clear cache)
            reload_data_and_clear_cache()
            return {"status": "success", "message": "Settings updated and data reloaded"}
        else:
             raise HTTPException(status_code=500, detail="Failed to save settings to file")

    except Exception as e:
        logging.error(f"Error updating settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk_metrics")
async def get_risk_metrics(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns portfolio risk metrics (Sharpe, Volatility, Max Drawdown).
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, original_csv_path, _ = data
    if df.empty:
        return {}

    try:
        # Calculate daily history to get the total portfolio value series
        daily_df, _, _, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=date(2000, 1, 1),
            end_date=date.today(),
            display_currency=currency,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            include_accounts=accounts,
            benchmark_symbols_yf=[], # Add empty benchmarks for risk metrics
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            interval="D",
            original_csv_file_path=original_csv_path
        )
        
        if daily_df is None or "Portfolio Value" not in daily_df.columns:
            return {}

        portfolio_values = daily_df["Portfolio Value"]
        metrics = calculate_all_risk_metrics(portfolio_values)
        return clean_nans(metrics)
    except Exception as e:
        logging.error(f"Error calculating risk metrics: {e}")
        return {"error": str(e)}

@router.get("/attribution")
async def get_attribution(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns performance attribution by sector and stock.
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, _, _ = data
    if df.empty:
        return {}

    try:
        # Get current summary rows which contain gains and sector info
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            data=data,
            current_user=current_user
        )
        
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
            return {"sectors": [], "stocks": []}
        
        rows = summary_df.to_dict(orient="records")

        # Sector Attribution
        sector_data = defaultdict(lambda: {"gain": 0.0, "value": 0.0})
        stock_data = []

        total_gain = 0.0
        for row in rows:
            symbol = row.get("Symbol")
            if symbol == "Total" or row.get("is_total"):
                continue
            
            gain_col = f"Total Gain ({currency})"
            value_col = f"Market Value ({currency})"
            
            gain = row.get(gain_col, 0.0)
            value = row.get(value_col, 0.0)
            sector = row.get("Sector") or "Unknown"
            
            total_gain += gain
            sector_data[sector]["gain"] += gain
            sector_data[sector]["value"] += value
            
            # --- AGGREGATION LOGIC ---
            # Group by Name if available to merge tickers (e.g. GOOG/GOOGL), otherwise by symbol
            name = row.get("Name")
            found = False
            for prev_item in stock_data:
                # Check for match by Name (if valid) OR Symbol
                # match if names are identical strings (and not None/empty)
                name_match = (name and prev_item.get("name") and name == prev_item.get("name"))
                symbol_match = (prev_item["symbol"] == symbol)

                if name_match or symbol_match:
                     prev_item["gain"] += gain
                     prev_item["value"] += value
                     
                     # If it was a name match but different symbol, merge the symbol string
                     if not symbol_match:
                         # Split existing symbols to check for uniqueness
                         existing_syms = [s.strip() for s in prev_item["symbol"].split(",")]
                         if symbol not in existing_syms:
                             prev_item["symbol"] += f", {symbol}"
                     
                     found = True
                     break
            
            if not found:
                 stock_data.append({
                    "symbol": symbol,
                    "name": name,
                    "gain": gain,
                    "value": value,
                    "sector": sector
                })


        # Format sector output
        sector_attribution = []
        for sector, d in sector_data.items():
            sector_attribution.append({
                "sector": sector,
                "gain": d["gain"],
                "value": d["value"],
                "contribution": (d["gain"] / total_gain if total_gain != 0 else 0)
            })

        # Sort by contribution
        sector_attribution.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        stock_data.sort(key=lambda x: abs(x["gain"]), reverse=True)
        
        # Calculate contribution % for stocks
        for stock in stock_data:
            stock["contribution"] = (stock["gain"] / total_gain) if total_gain != 0 else 0.0

        return clean_nans({
            "sectors": sector_attribution,
            "stocks": stock_data[:10], # Top 10 contributors/detractors
            "total_gain": total_gain
        })
    except Exception as e:
        logging.error(f"Error calculating attribution: {e}")
        return {"error": str(e)}

# --- HELPER: Average Daily Balance Calculation ---
def calculate_mtd_average_daily_balance(
    current_cash: float, 
    mtd_transactions: pd.DataFrame, 
    today_date: date
) -> float:
    """
    Calculates the Month-To-Date (MTD) Average Daily Balance (ADB).
    Reconstructs daily balances backwards from the current cash balance.
    """
    if current_cash == 0 and mtd_transactions.empty:
        return 0.0

    start_of_month = date(today_date.year, today_date.month, 1)
    days_in_month_so_far = (today_date - start_of_month).days + 1
    
    # Map dates to net change (assuming 'Total Amount' is signed correctly: + for inflow, - for outflow)
    # NOTE: In Investa, 'Total Amount' for BUY is negative (cash outflow), SELL is positive (cash inflow).
    # DEPOSIT is positive, WITHDRAWAL is negative.
    # So 'Total Amount' directly represents the change in cash balance.
    
    changes_by_date = {}
    if not mtd_transactions.empty:
        # Group by Date
        # Ensure Date column is datetime/date
        changes_by_date = mtd_transactions.groupby(mtd_transactions["Date"].dt.date)["Total Amount"].sum().to_dict()

    daily_balances = []
    
    # We walk BACKWARDS from Today
    running_balance = current_cash
    
    for d_idx in range(days_in_month_so_far):
        # Current day we are looking at (going backwards: Today, Yesterday, ...)
        lookback_date = today_date - timedelta(days=d_idx)
        
        # The running_balance represents the END OF DAY balance for lookback_date
        daily_balances.append(running_balance)
        
        # Before moving to yesterday (next iteration), adjust balance to get the start of day (end of previous day)
        # Start + Change = End  => Start = End - Change
        change_on_day = changes_by_date.get(lookback_date, 0.0)
        
        running_balance = running_balance - change_on_day
        
    if not daily_balances:
        return 0.0
        
    return sum(daily_balances) / len(daily_balances)

@router.get("/dividend_calendar")
async def get_dividend_calendar(
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    config_manager = Depends(get_config_manager),
    current_user: User = Depends(get_current_user)
):
    """
    Returns confirmed AND estimated dividend events for the next 12 months.
    """
    df, _, user_symbol_map, user_excluded_symbols, _, _, _ = data
    if df.empty:
        return []

    try:
        # Get current holdings symbols
        summary_data = await _calculate_portfolio_summary_internal(
            include_accounts=accounts, 
            show_closed_positions=False, 
            data=data,
            current_user=current_user
        )
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
             return []
        rows = summary_df.to_dict(orient="records")
        
        # Map Symbol -> Quantity
        holdings = defaultdict(float)
        for r in rows:
            sym = r["Symbol"]
            if sym != "Total" and not r.get("is_total"):
                 qty = r.get("Quantity", 0)
                 if qty > 0:
                     holdings[sym] += qty
        
        if not holdings:
            return []

        # Use MarketDataProvider to get basic info efficiently
        from market_data import MarketDataProvider, map_to_yf_symbol, _run_isolated_fetch
        from finutils import is_cash_symbol
        
        provider = get_mdp()
        yf_map = {} # Internal -> YF
        yf_symbols = set()
        
        for sym in holdings.keys():
            if is_cash_symbol(sym):
                continue
            yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols)
            if yf_sym:
                yf_map[sym] = yf_sym
                yf_symbols.add(yf_sym)

        # Fetch Fundamentals & Calendar in Parallel using ThreadPoolExecutor
        # This fixes the timeout issue of serial fetching while avoiding the data loss issue of batch fetching
        import yfinance as yf
        from datetime import timedelta
        import pandas as pd
        import concurrent.futures

        calendar_events = []
        today = date.today()
        end_date = today + timedelta(days=365) # 1 Year Projection
        
        def fetch_symbol_data(sym):
            """Helper to fetch data for a single symbol independently."""
            yf_sym = yf_map.get(sym)
            if not yf_sym: return []
            
            local_events = []
            qty = holdings.get(sym, 0)
            
            try:
                # Use MarketDataProvider cache for fundamentals (lighter and cached)
                info = provider.get_fundamental_data(yf_sym) or {}
                
                div_rate = info.get("trailingAnnualDividendRate", 0.0)
                last_div_val = info.get("lastDividendValue")
                if not div_rate: div_rate = info.get("dividendRate", 0.0)
                
                # 1. Confirmed Events (Need live ticker for calendar, but wrap carefully)
                # Only check calendar if we have indication of dividends
                if div_rate > 0 or last_div_val:
                    # Use isolated fetch for calendar data
                    cal = _run_isolated_fetch([yf_sym], task="calendar")
                    if cal and 'Dividend Date' in cal:
                        div_date_raw = cal['Dividend Date']
                        # Dates were stringified in the worker
                        if isinstance(div_date_raw, str):
                            try:
                                c_date = datetime.fromisoformat(div_date_raw).date()
                            except ValueError:
                                # Fallback if it's just 'YYYY-MM-DD'
                                try:
                                    c_date = datetime.strptime(div_date_raw, "%Y-%m-%d").date()
                                except ValueError:
                                    c_date = None
                        else:
                            c_date = div_date_raw
                            if isinstance(c_date, datetime): c_date = c_date.date()
                        
                        if c_date and c_date >= today:
                            amt = last_div_val if last_div_val else (div_rate / 4 if div_rate else 0)
                            local_events.append({
                                "symbol": sym,
                                "dividend_date": str(c_date),
                                "ex_dividend_date": str(cal.get('Ex-Dividend Date', '')),
                                "amount": amt * qty,
                                "status": "confirmed"
                            })
                
                # 2. Estimated Events
                if div_rate and div_rate > 0:

                    freq_months = 3
                    if last_div_val and last_div_val > 0:
                        ratio = div_rate / last_div_val
                        if 10 <= ratio <= 14: freq_months = 1
                        elif 3.5 <= ratio <= 5.5: freq_months = 3
                        elif 1.5 <= ratio <= 2.5: freq_months = 6
                        elif 0.8 <= ratio <= 1.2: freq_months = 12
                    
                    # Anchor
                    anchor = None
                    if local_events:
                        try:
                            anchor = datetime.strptime(local_events[-1]["dividend_date"], "%Y-%m-%d").date()
                        except: pass
                    
                    if not anchor and info.get("lastDividendDate"):
                        try:
                            anchor = date.fromtimestamp(info.get("lastDividendDate"))
                        except: pass
                        
                    if not anchor and info.get("exDividendDate"):
                        try:
                            anchor = date.fromtimestamp(info.get("exDividendDate")) + timedelta(days=21)
                        except: pass
                        
                    if anchor:
                        curr = anchor
                        while curr < today:
                            curr = (pd.Timestamp(curr) + pd.DateOffset(months=freq_months)).date()
                        
                        while curr <= end_date:
                            is_dup = False
                            for ce in local_events:
                                if ce["status"] == "confirmed":
                                    ce_date = datetime.strptime(ce["dividend_date"], "%Y-%m-%d").date()
                                    if abs((ce_date - curr).days) < 20: 
                                        is_dup = True
                                        break
                            
                            if not is_dup and curr >= today:
                                est_amt = (last_div_val if last_div_val else div_rate/4) * qty
                                local_events.append({
                                    "symbol": sym,
                                    "dividend_date": str(curr),
                                    "ex_dividend_date": "",
                                    "amount": est_amt,
                                    "status": "estimated"
                                })
                            
                            curr = (pd.Timestamp(curr) + pd.DateOffset(months=freq_months)).date()
                            
            except Exception as e:
                logging.warning(f"Error fetching data for {sym}: {e}")
                
            return local_events

        # Run in parallel
        # Run SEQUENTIALLY to absolutely prevent OOM/Thread exhaustion
        # This endpoint is a background feature, it can be slow.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_sym = {executor.submit(fetch_symbol_data, sym): sym for sym in holdings.keys()}
            for future in concurrent.futures.as_completed(future_to_sym):
                try:
                    events = future.result()
                    calendar_events.extend(events)
                except Exception as exc:
                    logging.error(f"Symbol generated an exception: {exc}")

        # Sort by date
        calendar_events.sort(key=lambda x: x["dividend_date"])
        
        # --- NEW: Add Virtual Interest on Cash via Helper ---
        try:
            config_manager.load_manual_overrides()
            interest_rates = config_manager.manual_overrides.get("account_interest_rates", {})
            thresholds = config_manager.manual_overrides.get("interest_free_thresholds", {})
            
            if interest_rates:
                cash_events = generate_cash_interest_events( # Changed to use imported function
                    df=df,
                    interest_rates=interest_rates,
                    thresholds=thresholds,
                    start_date=today,
                    end_date=end_date
                )
                calendar_events.extend(cash_events)
                
                # Re-sort after adding interest
                calendar_events.sort(key=lambda x: x["dividend_date"])
                
        except Exception as e:
            logging.error(f"Error adding cash interest: {e}")

        return clean_nans(calendar_events)
        
        return clean_nans(calendar_events)

    except Exception as e:
        logging.error(f"Error fetching dividend calendar: {e}", exc_info=True)
        return {"error": str(e)}

# --- FUNDAMENTALS & FINANCIALS ENDPOINTS ---

@router.get("/fundamentals/{symbol}")
async def get_fundamentals_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns fundamental data (ticker.info) for a symbol."""
    (_, _, user_symbol_map, user_excluded_symbols, _, _, _) = data
    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        fundamental_data = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        if fundamental_data is None:
             raise HTTPException(status_code=404, detail=f"No fundamental data found for {yf_symbol}")
        return clean_nans(fundamental_data)
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {yf_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/financials/{symbol}")
async def get_financials_endpoint(
    symbol: str,
    period_type: str = "annual",
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns historical financial statements for a symbol."""
    (_, _, user_symbol_map, user_excluded_symbols, _, _, _) = data
    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        financials = mdp.get_financials(yf_symbol, period_type, force_refresh=force)
        balance_sheet = mdp.get_balance_sheet(yf_symbol, period_type, force_refresh=force)
        cashflow = mdp.get_cashflow(yf_symbol, period_type, force_refresh=force)
        
        # Convert DataFrames to dicts for JSON serialization
        def df_to_dict(df):
            if df is None or df.empty: return {}
            return json.loads(df.to_json(orient="split", date_format="iso"))

        # Extract Shareholders' Equity from Balance Sheet if possible
        equity_items = [
            "Stockholders Equity", "Total Equity Gross Minority Interest", 
            "Common Stock Equity", "Retained Earnings", "Capital Stock", 
            "Common Stock", "Other Equity Adjustments", 
            "Gains Losses Not Affecting Retained Earnings",
            "Treasury Shares Number", "Ordinary Shares Number", "Share Issued"
        ]
        shareholders_equity = None
        if balance_sheet is not None and not balance_sheet.empty:
            # Filter rows that exist in the balance sheet index
            existing_equity_items = [item for item in equity_items if item in balance_sheet.index]
            if existing_equity_items:
                shareholders_equity = balance_sheet.loc[existing_equity_items]

        return clean_nans({
            "financials": df_to_dict(financials),
            "balance_sheet": df_to_dict(balance_sheet),
            "cashflow": df_to_dict(cashflow),
            "shareholders_equity": df_to_dict(shareholders_equity)
        })
    except Exception as e:
        logging.error(f"Error fetching financials for {yf_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ratios/{symbol}")
async def get_ratios_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns calculated financial ratios for a symbol."""
    if not FINANCIAL_RATIOS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Financial ratios module not available.")

    (_, _, user_symbol_map, user_excluded_symbols, _, _, _) = data
    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        # Fetch data needed for ratios
        info = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        financials = mdp.get_financials(yf_symbol, "annual", force_refresh=force)
        balance_sheet = mdp.get_balance_sheet(yf_symbol, "annual", force_refresh=force)
        
        # Calculate historical ratios
        historical_ratios_df = calculate_key_ratios_timeseries(financials, balance_sheet)
        
        # Calculate current valuation ratios
        current_valuation = calculate_current_valuation_ratios(info, financials, balance_sheet)
        
        # Format historical ratios
        def df_to_dict(df):
            if df is None or df.empty: return {}
            # Reset index to include 'Period'
            df_reset = df.reset_index()
            if 'Period' in df_reset.columns:
                df_reset['Period'] = df_reset['Period'].astype(str)
            return df_reset.to_dict(orient="records")

        return clean_nans({
            "historical": df_to_dict(historical_ratios_df),
            "valuation": current_valuation
        })
    except Exception as e:
        logging.error(f"Error calculating ratios for {yf_symbol}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intrinsic_value/{symbol}")
async def get_intrinsic_value_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    config_manager: ConfigManager = Depends(get_config_manager),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Returns calculated intrinsic value results for a symbol."""
    logging.info(f"CALCULATING INTRINSIC VALUE FOR {symbol} - CODE VERSION: 1.1 (CAPS ENABLED)")
    if not FINANCIAL_RATIOS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Financial ratios module not available.")

    (_, _, user_symbol_map, user_excluded_symbols, _, _, _) = data
    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        results = get_intrinsic_value_for_symbol(symbol, mdp, config_manager, force_refresh=force)
        
        if "error" in results:
             raise HTTPException(status_code=500, detail=results["error"])
        
        # We still need info for the sync function below
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
        info = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        
        # Sync to screener cache
        try:
            # db_conn injected
            if db_conn:
                if info:
                    info["valuation_details"] = results
                update_intrinsic_value_in_cache(
                    db_conn,
                    symbol,
                    results.get("average_intrinsic_value"),
                    results.get("margin_of_safety_pct"),
                    info.get("lastFiscalYearEnd"),
                    info.get("mostRecentQuarter"),
                    info=info
                )
        except Exception as e_sync:
            logging.warning(f"Failed to sync intrinsic value to cache for {symbol}: {e_sync}")

        return clean_nans(results)
    except Exception as e:
        logging.error(f"Error calculating intrinsic value for {yf_symbol}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# --- Phase 5: Settings & Webhook Endpoints ---

from server.dependencies import get_config_manager, reload_data
from config_manager import ConfigManager
from pydantic import BaseModel

class ManualOverrideRequest(BaseModel):
    symbol: str
    price: Optional[float] = None
    # Add other fields as needed for future (sector, etc.)

@router.post("/settings/manual_overrides")
async def update_manual_override(
    override: ManualOverrideRequest,
    current_user: User = Depends(get_current_user),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """
    Updates or removes a manual price override for a symbol.
    If price is None, removes the override.
    """
    try:
        # Load latest overrides
        config_manager.load_manual_overrides()
        
        symbol_upper = override.symbol.strip().upper()
        current_overrides = config_manager.manual_overrides.get("manual_price_overrides", {})
        
        if override.price is not None:
            # Add/Update
            # We preserve existing extra fields if any (like date), or create new dict
            existing_entry = current_overrides.get(symbol_upper, {})
            # If strictly setting price, update it.
            # Use 'manual' as source hint
            existing_entry["price"] = override.price
            existing_entry["source"] = "User Manual Override"
            existing_entry["updated_at"] = datetime.now().isoformat()
            
            # Ensure price is valid positive number or 0
            if override.price < 0:
                 raise HTTPException(status_code=400, detail="Price must be non-negative.")

            current_overrides[symbol_upper] = existing_entry
        else:
            # Remove override if it exists
            if symbol_upper in current_overrides:
                del current_overrides[symbol_upper]
        
        config_manager.manual_overrides["manual_price_overrides"] = current_overrides
        success = config_manager.save_manual_overrides()
        
        if not success:
             raise HTTPException(status_code=500, detail="Failed to save manual overrides file.")

        # --- Legacy Clean-up: Also remove from gui_config if present to avoid conflicts ---
        # Because dependencies.py merges them, we must ensure it's gone from legacy config too.
        gui_config_changed = False
        gc = config_manager.gui_config
        
        # Check both possible legacy keys
        for key in ["manual_price_overrides", "manual_overrides_dict"]:
            if key in gc and isinstance(gc[key], dict):
                if symbol_upper in gc[key]:
                    del gc[key][symbol_upper]
                    gui_config_changed = True
        
        # Also if we are Adding, we might want to Add to legacy? 
        # No, let's strictly migrate to JSON. But we must ensure Legacy doesn't overwrite.
        # If we Add to JSON, dependencies.py (with my previous fix) will Update Legacy with JSON.
        # So JSON wins for Add/Update.
        # But for Delete, JSON just lacks the key, so Legacy wins.
        # So we MUST delete from Legacy.
        
        if gui_config_changed:
             config_manager.save_gui_config()
             logging.info(f"Removed {symbol_upper} from legacy gui_config.")
        # -------------------------------------------------------------------------------

        # Force reload of data so next request uses new override
        reload_data_and_clear_cache()
        
        return {"status": "success", "message": f"Override for {symbol_upper} updated."}

    except Exception as e:
        logging.error(f"Error updating manual override: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class WebhookRefreshRequest(BaseModel):
    secret: str

@router.get("/portfolio_health")
async def get_portfolio_health(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user)
):
    """
    Returns a comprehensive portfolio health score and breakdown.
    """
    try:
        logging.info(f"Health: Fetching summary for accounts: {accounts}")
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=False,
            data=data,
            current_user=current_user
        )
        summary_df = summary_data.get("summary_df")
        
        if summary_df is None:
            logging.warning("Health: Summary DF is None")
            summary_df = pd.DataFrame()
        else:
            logging.info(f"Health: Summary DF shape: {summary_df.shape}")
        
        # 2. Get Risk Metrics (for efficiency/volatility)
        logging.info("Health: Fetching history (1y period)")
        history_df = _calculate_historical_performance_internal(
            currency=currency,
            period="1y", # Standard period for health check
            accounts=accounts,
            benchmarks=None,
            data=data,
            return_df=True # Requested DF for calculations
        )
        
        portfolio_series = pd.Series(dtype=float)
        if history_df is not None and not history_df.empty and "value" in history_df.columns:
             logging.info(f"Health: History DF shape: {history_df.shape}")
             # Extract portfolio portfolio value series
             portfolio_series = history_df.set_index("date")["value"]
        else:
             logging.warning(f"Health: History DF is empty or missing 'value'. Columns: {history_df.columns if history_df is not None else 'None'}")
             
        risk_metrics = calculate_all_risk_metrics(portfolio_series)
        logging.info(f"Health: Risk Metrics: {risk_metrics}")
        
        # 3. Calculate Health Score
        from portfolio_analyzer import calculate_health_score
        health = calculate_health_score(summary_df, risk_metrics)
        logging.info(f"Health: Final Health Score: {health.get('overall_score')}")
        
        return health

    except Exception as e:
        logging.error(f"Error calculating portfolio health: {e}", exc_info=True)
        # Return a safe default instead of crashing, but include error for debugging
        return {
            "overall_score": 0,
            "rating": "Error",
            "debug_error": str(e), # Temporary for debugging
            "components": {
                "diversification": {"score": 0, "metric": 0, "label": f"Err: {str(e)[:20]}"},
                "efficiency": {"score": 0, "metric": 0, "label": "Error"},
                "stability": {"score": 0, "metric": "0%", "label": "Error"}
            }
        }

@router.post("/webhook/refresh")
async def webhook_refresh(
    request: WebhookRefreshRequest
):
    """
    Webhook to trigger a market data refresh (cache invalidation).
    Requires a shared secret.
    """
    # Simple hardcoded check for now, or load from env/config
    # For personal local app, a default simple secret is acceptable if not exposed to internet
    # Ideally should be in config.py or env var.
    EXPECTED_SECRET = os.environ.get("INVESTA_WEBHOOK_SECRET", "investa_refresh_secret_123")
    
    if request.secret.strip() != EXPECTED_SECRET:
        logging.warning(f"Webhook Secret Mismatch. Input: '{request.secret}' != Expected: (hidden)")
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        # 1. Invalidate Market Cache
        # The cache file is typically DEFAULT_CURRENT_CACHE_FILE_PATH
        # We can either delete it or rely on MarketDataProvider to manage it.
        # Safest is to delete the file.
        
        app_data_dir = config.get_app_data_dir()
        cache_path = os.path.join(app_data_dir, config.DEFAULT_CURRENT_CACHE_FILE_PATH)
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logging.info(f"Webhook: Deleted market data cache at {cache_path}")
        else:
             logging.info("Webhook: Cache file not found (already clean).")
             
        # 2. Reload internal transaction cache
        reload_data_and_clear_cache()
        
        return {"status": "success", "message": "Market data cache invalidated and data reloaded."}
        
    except Exception as e:
        logging.error(f"Error in webhook refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- WATCHLIST ENDPOINTS ---

class WatchlistAdd(BaseModel):
    symbol: str
    note: Optional[str] = ""
    watchlist_id: Optional[int] = 1

class WatchlistCreate(BaseModel):
    name: str

class WatchlistRename(BaseModel):
    name: str

@router.get("/watchlists")
async def get_watchlists_endpoint(
    current_user: User = Depends(get_current_user), 
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Fetches all available watchlists."""
    # conn injected
    return get_all_watchlists(conn)

@router.post("/watchlists")
async def create_watchlist_endpoint(
    item: WatchlistCreate, 
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Creates a new watchlist."""
    # conn injected
    new_id = create_watchlist(conn, item.name, user_id=current_user.id)
    if not new_id:
            raise HTTPException(status_code=500, detail="Failed to create watchlist")
    return {"id": new_id, "name": item.name}

@router.put("/watchlists/{watchlist_id}")
async def rename_watchlist_endpoint(
    watchlist_id: int, 
    item: WatchlistRename, 
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Renames a watchlist."""
    # conn injected
    success = rename_watchlist(conn, watchlist_id, item.name)
    if not success:
            raise HTTPException(status_code=404, detail="Watchlist not found or failed to rename")
    return {"status": "success"}

@router.delete("/watchlists/{watchlist_id}")
async def delete_watchlist_endpoint(
    watchlist_id: int, 
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Deletes a watchlist."""
    # conn injected
    success = delete_watchlist(conn, watchlist_id)
    if not success:
            raise HTTPException(status_code=404, detail="Watchlist not found or failed to delete")
    return {"status": "success"}

@router.get("/watchlist")
async def get_watchlist_endpoint(
    watchlist_id: int = Query(1, alias="id"),
    currency: str = "USD",
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """
    Fetches the watchlist enriched with current market prices.
    """
    (
        _,
        _,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        _,
        _
    ) = data

    # conn injected
    # Remove manual check
    pass
    
    try:
        # Verify ownership
        allowed = [w['id'] for w in get_all_watchlists(conn)]
        if watchlist_id not in allowed:
             if not allowed and watchlist_id == 1: 
                 pass 
             else:
                 return []

        db_items = get_watchlist(conn, watchlist_id=watchlist_id)
        if not db_items:
            return []
            
        mdp = get_mdp()
        
        # Extract symbols for batch fetching
        symbols = [item["Symbol"] for item in db_items]
        
        # Fetch current quotes
        try:
            # Correctly unpack 5 values: results, fx_rates_vs_usd, fx_prev_close_vs_usd, has_errors, has_warnings
            quotes, fx_rates, fx_prev, q_errors, q_warnings = mdp.get_current_quotes(
                internal_stock_symbols=symbols,
                required_currencies=set(),
                user_symbol_map=user_symbol_map,
                user_excluded_symbols=user_excluded_symbols
            )
        except Exception as e_quotes:
            logging.error(f"Failed to fetch quotes for watchlist: {e_quotes}")
            quotes = {}

        # Fetch fundamentals (Market Cap, PE, Yield)
        try:
            fundamentals = mdp.get_fundamentals_batch(
                symbols, 
                user_symbol_map=user_symbol_map, 
                user_excluded_symbols=user_excluded_symbols
            )
        except Exception as e_fund:
            logging.error(f"Failed to fetch fundamentals for watchlist: {e_fund}")
            fundamentals = {}

        enriched_items = []
        for item in db_items:
            symbol = item["Symbol"]
            quote = quotes.get(symbol, {})
            fund = fundamentals.get(symbol, {})
            
            # Sparkline data is already provided in the quote from get_current_quotes batch fetch
            sparkline = quote.get("sparkline_7d", [])

            enriched_items.append({
                **item,
                "Price": quote.get("price"),
                "Day Change": quote.get("change"),
                "Day Change %": quote.get("changesPercentage"),
                "Name": quote.get("name"),
                "Currency": quote.get("currency"),
                "Sparkline": sparkline,
                "Market Cap": fund.get("marketCap"),
                "PE Ratio": fund.get("trailingPE") or fund.get("forwardPE"),
                "Dividend Yield": fund.get("dividendYield")
            })
        
        return clean_nans(enriched_items)
    finally:
        # conn close handled by dependency
        pass

@router.post("/watchlist")
async def add_to_watchlist_api(
    item: WatchlistAdd, 
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Adds a symbol to the watchlist."""
    # conn injected
    try:
        symbol_upper = item.symbol.strip().upper()
        if not symbol_upper:
            raise HTTPException(status_code=400, detail="Symbol is required")
            
        # Use provided watchlist_id or default to 1
        wl_id = item.watchlist_id or 1
        
        # Verify ownership
        allowed = [w['id'] for w in get_all_watchlists(conn)]
        if wl_id not in allowed:
            # If user has no watchlists, and asks for 1? 
            # We strictly enforce that the watchlist must belong to user. 
            # If user has no watchlists, they should create one. 
            # BUT: Legacy behavior was default id=1. 
            # Migration created watchlist 1 for user 1. 
            # New users might not have watchlist 1. 
            # Frontend should create watchlist if needed.
            # We return 403.
            raise HTTPException(status_code=403, detail="Not authorized to modify this watchlist. Please create a watchlist first.")
            
        success = add_to_watchlist(conn, symbol_upper, item.note, watchlist_id=wl_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add to watchlist")
        return {"status": "success"}
    finally:
        # conn close handled by dependency
        pass

@router.delete("/watchlist/{symbol}")
async def remove_from_watchlist_api(
    symbol: str, 
    watchlist_id: int = Query(1, alias="id"), 
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Removes a symbol from the watchlist."""
    # conn injected
    try:
        # Verify ownership
        allowed = [w['id'] for w in get_all_watchlists(conn)]
        if watchlist_id not in allowed:
            raise HTTPException(status_code=403, detail="Not authorized to modify this watchlist")
            
        success = remove_from_watchlist(conn, symbol.strip().upper(), watchlist_id=watchlist_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove from watchlist")
        return {"success": True} # Legacy return format? Standardize to status success
    finally:
        # conn close handled by dependency
        pass
@router.post("/clear_cache")
async def clear_cache():
    """Clears all application caches (files and in-memory)."""
    try:
        logging.info("Starting Cache Clearing Process...")
        deleted_count = 0
        
        # 1. Clear In-Memory Caches
        _PORTFOLIO_SUMMARY_CACHE.clear()
        # Also try to clear MarketDataProvider's internal cache if possible (it re-reads file anyway)
        
        # 2. Identify Cache Directories
        app_data_dir = config.get_app_data_dir()
        app_cache_dir = config.get_app_cache_dir()
        
        targets = [app_data_dir]
        if app_cache_dir and app_cache_dir != app_data_dir:
            targets.append(app_cache_dir)
            
        # Removed broad system cache scanning for safety and performance.
        # We only want to clear OUR app's cache.
            
        logging.info(f"Target Directories for cleanup: {targets}")

        # Files/Dirs that MUST be deleted
        # Explicit filenames to target in ANY target directory
        EXPLICIT_FILES_TO_DELETE = {
            "portfolio_cache_yf.json",
            "yf_metadata_cache.json",
            "invalid_symbols_cache.json",
            "portfolio_cache.json", # Legacy name?
        }
        
        # Directory names to recursively delete
        CACHE_DIR_NAMES = {
            'historical_data_cache', 
            # 'fundamentals_cache', # Preserved by user request
            'all_holdings_cache_new',
            'daily_results_cache',
            'test_fx_cache' # Added this
        }
        
        # Extensions that imply cache (be careful not to delete config/overrides)
        CACHE_EXTENSIONS = ('.json', '.feather', '.npy', '.key') 
        
        # Safe-List (Never Delete)
        KEEP_FILES = {'gui_config.json', 'manual_overrides.json', 'investa_transactions.db'}
        KEEP_EXTENSIONS = ('.db', '.sqlite', '.sqlite3', '.bak')

        import shutil
        
        for base_dir in targets:
            if not os.path.exists(base_dir):
                continue
                
            logging.info(f"Scanning {base_dir}...")
            
            try:
                items = os.listdir(base_dir)
            except Exception as e:
                logging.warning(f"Could not list {base_dir}: {e}")
                continue

            for item in items:
                item_path = os.path.join(base_dir, item)
                
                # PROTECTED CHECKS
                if item in KEEP_FILES:
                    continue
                if any(item.lower().endswith(ext) for ext in KEEP_EXTENSIONS):
                    continue

                # A. Handle Subdirectories
                if os.path.isdir(item_path):
                    if item in CACHE_DIR_NAMES or item.startswith("yf_portfolio_hist"):
                        try:
                            # Count files inside for reporting
                            for _, _, files in os.walk(item_path):
                                deleted_count += len(files)
                            shutil.rmtree(item_path)
                            deleted_count += 1 
                            logging.info(f"Deleted Directory: {item}")
                        except Exception as e:
                            logging.warning(f"Failed to delete cache dir {item_path}: {e}")
                    continue
                
                # B. Handle Files
                if os.path.isfile(item_path):
                    should_delete = False
                    
                    # 1. Explicit Match
                    if item in EXPLICIT_FILES_TO_DELETE:
                        should_delete = True
                        
                    # 2. Prefix Match (High Confidence)
                    elif any(item.startswith(p) for p in [
                        config.HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, 
                        config.DAILY_RESULTS_CACHE_PATH_PREFIX,
                        "yf_portfolio_" # Catch-all for yf caches
                    ]):
                        should_delete = True
                        
                    # 3. Extension Match (Low Confidence - only in specific dirs)
                    # Only delete by extension if we are SURE it's a cache file
                    elif any(item.lower().endswith(ext) for ext in CACHE_EXTENSIONS):
                         # Extra safety: Don't delete random JSONs in app_data_dir unless they look like cache
                         if base_dir == app_data_dir and item.endswith(".json"):
                             if "cache" in item.lower():
                                 should_delete = True
                             else:
                                 should_delete = False # Skip unknown JSONs in config dir
                         else:
                             should_delete = True # In Caches/ folder, delete all JSONs/Feathers

                    if should_delete:
                        try:
                            os.remove(item_path)
                            deleted_count += 1
                            logging.info(f"Deleted File: {item}")
                        except Exception as e:
                            logging.warning(f"Failed to delete cache file {item_path}: {e}")
        
        # 3. Reload Data
        logging.info("Reloading data after cache clear...")
        reload_data_and_clear_cache()
        
        return {"status": "success", "message": f"Cache cleared. {deleted_count} items removed."}
    except Exception as e:
        logging.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class ScreenerRequest(BaseModel):
    universe_type: str = Field(..., description="watchlist, manual, or sp500")
    universe_id: Optional[str] = None
    manual_symbols: Optional[List[str]] = None
    fast_mode: Optional[bool] = False

@router.post("/screener/run")
async def run_screener(
    request: ScreenerRequest,
    current_user: User = Depends(get_current_user),
    data: tuple = Depends(get_transaction_data),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """
    Runs the stock screener based on the selected universe.
    """
    try:
        universe_type = request.universe_type
        universe_id = request.universe_id
        manual_symbols = request.manual_symbols
        
        # Handle "Holdings" as a dynamic universe if requested
        if universe_type == "holdings":
            summary_data = await _calculate_portfolio_summary_internal(
                data=data, 
                current_user=current_user, 
                show_closed_positions=False
            )
            summary_df = summary_data.get("summary_df")
            if summary_df is not None and not summary_df.empty:
                # Filter out "Total" row and get unique symbols
                manual_symbols = summary_df[~summary_df.get("is_total", False) & (summary_df["Symbol"] != "Total")]["Symbol"].tolist()
                # We treat it as a manual list once resolved
                universe_type = "manual" 

        results = await run_in_threadpool(screen_stocks, universe_type, universe_id, manual_symbols, db_conn=db_conn, fast_mode=request.fast_mode)
        return clean_nans(results)
    except Exception as e:
        logging.error(f"Screener error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/screener/review/{symbol}")
async def trigger_ai_review(
    symbol: str, 
    force: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """
    Triggers (or retrieves cached) AI review for a specific stock.
    """
    try:
        # We need to fetch data first to pass to AI
        mdp = get_mdp()
        # Ensure data exists
        quotes, _, _, _, _ = mdp.get_current_quotes([symbol], {"USD"}, {}, set())
        details = mdp.get_ticker_details_batch({symbol})
        
        fund_data = details.get(symbol, {})
        # Merge price
        if symbol in quotes:
            fund_data["currentPrice"] = quotes[symbol].get("price")
            
        if not fund_data:
             raise HTTPException(status_code=404, detail=f"Data not found for {symbol}")
        
        # Ratios data - we can pass empty or minimal if we don't have full ratios calculated.
        # AI Analyzer expects 'ratios_data' dict with keys like 'Return on Equity (ROE) (%)'
        # We can extract some from fund_data if available (trailingPE etc are in fund_data)
        
        # Map info keys to ratio keys - simplified for Screen context
        ratios_data = {
            "Return on Equity (ROE) (%)": fund_data.get("returnOnEquity", 0) * 100 if fund_data.get("returnOnEquity") else None,
            "Gross Profit Margin (%)": fund_data.get("grossMargins", 0) * 100 if fund_data.get("grossMargins") else None,
            "Net Profit Margin (%)": fund_data.get("profitMargins", 0) * 100 if fund_data.get("profitMargins") else None,
            "Debt-to-Equity Ratio": fund_data.get("debtToEquity", 0) / 100 if fund_data.get("debtToEquity") else None,
            "Current Ratio": fund_data.get("currentRatio"),
        }
        
        review = generate_stock_review(symbol, fund_data, ratios_data, force_refresh=force)
        return clean_nans(review)
        
    except Exception as e:
        logging.error(f"AI Review error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
