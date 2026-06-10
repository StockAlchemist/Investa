"""Portfolio summary/history calculation engine shared by the route modules.

Owns the in-process result caches (LRU, bounded) and the background
pre-calculation pool. Route modules import the public helpers; nothing here
imports from server.api, so route modules can depend on this freely.
"""

# ruff: noqa: E402
import asyncio
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, time as dt_time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

import config
from config import BENCHMARK_MAPPING, YFINANCE_INDEX_TICKER_MAP
from db_utils import get_db_connection
from portfolio_logic import (
    CURRENT_HIST_VERSION,
    calculate_historical_performance,
    calculate_portfolio_summary,
)
from server.auth import User
from server.dependencies import get_config_manager, reload_data
from server.route_utils import _lru_get, _lru_put, clean_nans, get_mdp
from server.routes.market import clear_market_history_cache
from utils_time import get_est_today, get_latest_trading_date, is_market_open

_PRECALC_POOL = ThreadPoolExecutor(max_workers=2)
_PRECALC_IN_FLIGHT: set = set()
_PORTFOLIO_SUMMARY_CACHE: OrderedDict = OrderedDict()
_RAW_CALC_CACHE: OrderedDict = OrderedDict()
_PORTFOLIO_HISTORY_CACHE: OrderedDict = OrderedDict()
_HISTORY_CALC_FUTURES = {}    # Track in-flight historical calculations
_SUMMARY_CALC_LOCK = asyncio.Lock() # Lock to prevent concurrent calculation on cache miss


def _user_db_path(username: str) -> str:
    """Reconstruct the per-user SQLite path from a username."""
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, username)
    return os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)


def _evict_user_summary_cache(username: str):
    """Remove only the given user's entries from _PORTFOLIO_SUMMARY_CACHE.

    The summary cache key has db_path at index 2, which is unique per user.
    """
    user_path = _user_db_path(username)
    stale = [k for k in _PORTFOLIO_SUMMARY_CACHE if k[2] == user_path]
    for k in stale:
        del _PORTFOLIO_SUMMARY_CACHE[k]


def _evict_user_history_cache(username: str):
    """Clear all portfolio history cache entries on a user write.

    History cache keys don't include username (they use db_mtime for isolation),
    so we conservatively clear the whole cache to avoid serving stale data.
    """
    _PORTFOLIO_HISTORY_CACHE.clear()


def reload_data_and_clear_cache(current_user: Optional[User] = None):
    """Clear only the current user's cached data, leaving other users' caches intact."""
    username = current_user.username if current_user else None
    reload_data(username)
    if username:
        _evict_user_summary_cache(username)
        _evict_user_history_cache(username)
        # Market history is shared benchmark data — safe to clear entirely on a write
        clear_market_history_cache()
    else:
        _PORTFOLIO_SUMMARY_CACHE.clear()
        clear_market_history_cache()
        _PORTFOLIO_HISTORY_CACHE.clear()
    logging.info(f"Caches cleared for user '{username or 'all'}'.")
    if current_user:
        trigger_background_precalculation(current_user)


def clear_portfolio_caches():
    """Clears calculated caches (Summary, History) without wiping the transaction dataframe cache."""
    _PORTFOLIO_SUMMARY_CACHE.clear()
    clear_market_history_cache()
    _PORTFOLIO_HISTORY_CACHE.clear()
    logging.info("Portfolio Summary, Market History, and Portfolio History caches cleared (Transaction cache retained).")


def trigger_background_precalculation(current_user: User):
    """Triggers background task to calculate and store portfolio snapshots."""
    if current_user.username in _PRECALC_IN_FLIGHT:
        logging.debug(f"Precalc already in flight for {current_user.username}, skipping duplicate submission.")
        return
    _PRECALC_IN_FLIGHT.add(current_user.username)

    def run_precalc():
        try:
            logging.info(f"Starting background metric pre-calculation for user {current_user.username}")
            from server.dependencies import get_transaction_data
            
            # retrieve user transaction data manually
            df, manual, user_map, excluded, acc_curr, cash_mode, path, mtime = get_transaction_data(current_user)
            if df.empty:
                logging.info("Skip precalc: dataframe is empty")
                return

            from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
            from config import DEFAULT_CURRENCY
            import config
            from datetime import date, datetime

            today = date.today()
            
            # Call synchronous summary generator
            overall_metrics, summary_df, holdings_dict, account_metrics, _, _, status = calculate_portfolio_summary(
                all_transactions_df_cleaned=df,
                original_transactions_df_for_ignored=df,
                ignored_indices_from_load=set(),
                ignored_reasons_from_load={},
                display_currency="USD",
                account_currency_map=acc_curr,
                default_currency=DEFAULT_CURRENCY,
                include_accounts=None, # ALL accounts
                manual_overrides_dict=manual,
                user_symbol_map=user_map,
                user_excluded_symbols=excluded,
                account_cash_mode_map=cash_mode  # account_cash_mode_map from get_transaction_data
            )

            # Calculate TWR synchronously for ALL account
            overall_twr = 0.0
            min_date = df["Date"].min().date()
            if not df.empty:
                _, _, _, hist_status = calculate_historical_performance(
                    all_transactions_df_cleaned=df,
                    original_transactions_df_for_ignored=df,
                    ignored_indices_from_load=set(),
                    ignored_reasons_from_load={},
                    start_date=min_date,
                    end_date=today,
                    interval="1d",
                    benchmark_symbols_yf=[],
                    display_currency="USD",
                    account_currency_map=acc_curr,
                    default_currency=DEFAULT_CURRENCY,
                    use_raw_data_cache=True,
                    use_daily_results_cache=True,
                    num_processes=None,
                    include_accounts=None,
                    worker_signals=None,
                    user_symbol_map=user_map,
                    manual_overrides_dict=manual,
                    user_excluded_symbols=excluded,
                    original_csv_file_path=path,
                    account_cash_mode_map=cash_mode,  # account_cash_mode_map
                    calc_method=None  # Use configured HISTORICAL_CALC_METHOD
                )
                if "|||TWR_FACTOR:" in hist_status:
                    try:
                        twr_part = hist_status.split("|||TWR_FACTOR:")[1].strip()
                        if twr_part != "NaN":
                            overall_twr = float(twr_part)
                    except (ValueError, IndexError) as e:
                        logging.debug(f"TWR factor parse error: {e}")
            
            # Note: For simplicity and performance, we only pre-calculate TWR for the overall portfolio right now.
            # We can calculate for individual accounts if needed.

            # Store in portfolio_snapshots table
            user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
            db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
            
            conn = get_db_connection(db_path, use_cache=False)
            if not conn:
                logging.error(f"Failed to connect to user database for precalculation: {db_path}")
                return

            def sf(val):
                if val is None:
                    return 0.0
                try:
                    import math
                    return 0.0 if math.isnan(float(val)) else float(val)
                except (TypeError, ValueError):
                    return 0.0

            try:
                # Use connection as context manager so any failure rolls back atomically.
                with conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM portfolio_snapshots WHERE snapshot_date=?', (today.isoformat(),))

                    if overall_metrics:
                        cursor.execute('''
                            INSERT INTO portfolio_snapshots (snapshot_date, account, total_value, total_cost, total_gain, total_return_pct, twr, irr, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            today.isoformat(),
                            'ALL',
                            sf(overall_metrics.get('market_value')),
                            sf(overall_metrics.get('total_buy_cost')),
                            sf(overall_metrics.get('total_gain')),
                            sf(overall_metrics.get('total_return_pct')),
                            sf(overall_twr),
                            sf(overall_metrics.get('portfolio_mwr')),
                            datetime.now().isoformat()
                        ))

                    if account_metrics:
                        for acc, acc_data in account_metrics.items():
                            cursor.execute('''
                                INSERT INTO portfolio_snapshots (snapshot_date, account, total_value, total_cost, total_gain, total_return_pct, twr, irr, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                today.isoformat(),
                                acc,
                                sf(acc_data.get('total_market_value_display')),
                                sf(acc_data.get('total_buy_cost_display')),
                                sf(acc_data.get('total_gain_display')),
                                sf(acc_data.get('total_return_pct')),
                                sf(acc_data.get('twr', 0.0)),
                                sf(acc_data.get('mwr')),
                                datetime.now().isoformat()
                            ))
            finally:
                conn.close()
            logging.info(f"Finished background metric pre-calculation for user {current_user.username}")
        except Exception as e:
            logging.error(f"Error in background metric pre-calculation: {e}", exc_info=True)
        finally:
            _PRECALC_IN_FLIGHT.discard(current_user.username)

    _PRECALC_POOL.submit(run_precalc)


async def _get_historical_performance_cached(
    df: pd.DataFrame,
    manual_overrides_dict: Dict,
    user_symbol_map: Dict,
    user_excluded_symbols: set,
    account_currency_map: Dict,
    original_csv_file_path: Optional[str],
    start_date: date,
    end_date: date,
    interval: str,
    benchmark_symbols_yf: List[str],
    display_currency: str,
    include_accounts: Optional[List[str]],
    account_cash_mode_map: Dict[str, str], # NEW
    db_mtime: float
) -> Tuple[pd.DataFrame, Dict, Dict, str]:
    """
    Wrapper for calculate_historical_performance that uses a shared in-memory cache
    and implements concurrency control (task sharing) to prevent the thundering herd effect.
    """
    # Create a unique key for the request parameters and data state
    accounts_key = tuple(sorted(include_accounts)) if include_accounts else "ALL"
    benchmarks_key = tuple(sorted(benchmark_symbols_yf)) if benchmark_symbols_yf else ()
    
    # We bucket market data freshness to 5 minutes if market is open, or 1 hour if closed
    from utils_time import is_market_open
    if is_market_open():
        time_bucket = int(time.time() / (5 * 60)) # 5 mins
    else:
        time_bucket = int(time.time() / (60 * 60)) # 1 hour
        
    cache_key = (
        display_currency,
        accounts_key,
        benchmarks_key,
        start_date,
        end_date,
        interval,
        db_mtime,
        time_bucket,
        CURRENT_HIST_VERSION
    )
    
    # 1. Check if we already have a cached result
    cached = _lru_get(_PORTFOLIO_HISTORY_CACHE, cache_key)
    if cached is not None:
        logging.info(f"Using cached Portfolio History for key: {cache_key[:3]}...")
        return cached
        
    # 2. Check if another request is already calculating this
    if cache_key in _HISTORY_CALC_FUTURES:
        logging.info(f"Historical calculation in progress for key {cache_key[:3]}... waiting.")
        try:
            return await _HISTORY_CALC_FUTURES[cache_key]
        except Exception as e:
            # If the future failed, we'll try to re-calculate (though usually better to just re-raise)
            logging.error(f"Waiting for historical calculation failed: {e}")
            raise
        
    # 3. No cache and no in-flight request, so we calculate
    future = asyncio.Future()
    _HISTORY_CALC_FUTURES[cache_key] = future
    
    try:
        logging.info(f"Starting historical performance calculation for key {cache_key[:3]}...")
        
        # Since calculate_historical_performance is CPU-bound, run in threadpool
        def run_calc():
            return calculate_historical_performance(
                all_transactions_df_cleaned=df,
                original_transactions_df_for_ignored=df,
                ignored_indices_from_load=set(),
                ignored_reasons_from_load={},
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                benchmark_symbols_yf=benchmark_symbols_yf,
                display_currency=display_currency,
                account_currency_map=account_currency_map,
                default_currency=config.DEFAULT_CURRENCY,
                include_accounts=include_accounts,
                manual_overrides_dict=manual_overrides_dict,
                user_symbol_map=user_symbol_map,
                user_excluded_symbols=user_excluded_symbols,
                original_csv_file_path=original_csv_file_path,
                account_cash_mode_map=account_cash_mode_map # PASSING IT HERE
            )
            
        result = await run_in_threadpool(run_calc)
        _lru_put(_PORTFOLIO_HISTORY_CACHE, cache_key, result)
        future.set_result(result)
        return result
    except Exception as e:
        if not future.done():
            future.set_exception(e)
        logging.error(f"Error in historical calculation: {e}", exc_info=True)
        raise
    finally:
        # Remove from in-flight tracker
        _HISTORY_CALC_FUTURES.pop(cache_key, None)


def _filter_closed_positions(result: Dict[str, Any], show_closed_positions: bool) -> Dict[str, Any]:
    """Filter out closed positions from a portfolio summary result if requested.
    
    The cache always stores results with show_closed_positions=True (BN-01).
    This helper applies the filter on cache retrieval when the caller requests
    show_closed_positions=False.
    """
    if show_closed_positions:
        return result
    
    sdf = result.get("summary_df")
    if sdf is None or (hasattr(sdf, 'empty') and sdf.empty):
        return result
    
    from portfolio_logic import STOCK_QUANTITY_CLOSE_TOLERANCE
    
    # Only filter if the DataFrame has a Quantity column
    if hasattr(sdf, 'columns') and "Quantity" in sdf.columns:
        mask = (
            (sdf["Quantity"].abs() >= STOCK_QUANTITY_CLOSE_TOLERANCE) |
            (sdf.get("is_total", False)) |
            (sdf["Symbol"] == "Total")
        )
        filtered_result = result.copy()
        filtered_result["summary_df"] = sdf[mask].copy()
        return filtered_result
    
    return result


def compute_account_closure_state(
    include_accounts: Optional[List[str]],
    closure_dates: Dict[str, str],
    today_date: date,
) -> Tuple[List[str], bool]:
    """Given a slice of accounts + the user's account_closure_dates map, return
    (closed_in_slice, all_selected_closed). `all_selected_closed` is True only
    when every account in `include_accounts` has a closure date <= today_date.

    Unparseable date strings are ignored (treated as open). An empty / None
    `include_accounts` (the all-accounts view) is never considered closed.
    """
    closed_in_slice: List[str] = []
    if not include_accounts:
        return closed_in_slice, False
    for acc in include_accounts:
        d_str = closure_dates.get(acc)
        if not d_str:
            continue
        try:
            closure_date = datetime.strptime(str(d_str), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if closure_date <= today_date:
            closed_in_slice.append(acc)
    all_selected_closed = (
        bool(closed_in_slice) and len(closed_in_slice) == len(include_accounts)
    )
    return closed_in_slice, all_selected_closed


async def _compute_raw_summary(
    currency: str,
    include_accounts: Optional[List[str]],
    data: tuple,
    account_interest_rates: dict,
    interest_free_thresholds: dict,
):
    """Run (and cache) the heavy calculate_portfolio_summary.

    Returns ``(overall_summary_metrics, summary_df, holdings_dict,
    account_level_metrics)``. The result is cached in ``_RAW_CALC_CACHE`` and
    shared between ``/summary`` and ``/summary/headline`` so the expensive math
    runs at most once per cache window. Always computes with
    ``show_closed_positions=True`` so a single cache entry serves both views.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
        db_path,
        db_mtime,
    ) = data

    accounts_key = tuple(sorted(include_accounts)) if include_accounts else "ALL"
    cache_ttl_seconds = 60 if is_market_open() else 300
    time_key = int(time.time() / cache_ttl_seconds)
    raw_key = (currency, accounts_key, db_path, db_mtime, time_key)

    cached = _lru_get(_RAW_CALC_CACHE, raw_key)
    if cached is not None:
        return cached

    async with _SUMMARY_CALC_LOCK:
        # Another request may have computed it while we waited for the lock.
        cached = _lru_get(_RAW_CALC_CACHE, raw_key)
        if cached is not None:
            return cached

        logging.info(f"Acquired lock. Calculating raw summary. Time key: {time_key}")
        mdp = get_mdp()

        # Offload heavy synchronous calculation to threadpool
        from fastapi.concurrency import run_in_threadpool

        def run_calc():
            return calculate_portfolio_summary(
                all_transactions_df_cleaned=df,
                original_transactions_df_for_ignored=df,
                ignored_indices_from_load=set(),
                ignored_reasons_from_load={},
                fmp_api_key=getattr(config, "FMP_API_KEY", None),
                display_currency=currency,
                show_closed_positions=True,  # ALWAYS compute all for cache sharing (BN-01)
                manual_overrides_dict=manual_overrides,
                user_symbol_map=user_symbol_map,
                user_excluded_symbols=user_excluded_symbols,
                include_accounts=include_accounts,
                account_currency_map=account_currency_map,
                default_currency=config.DEFAULT_CURRENCY,
                market_provider=mdp,
                account_interest_rates=account_interest_rates,
                interest_free_thresholds=interest_free_thresholds,
                account_cash_mode_map=account_cash_mode_map,
                db_mtime=db_mtime,  # BN-08: pass db_mtime for FIFO cache
            )

        try:
            (
                overall_summary_metrics,
                summary_df,
                holdings_dict,
                account_level_metrics,
                _,
                _,
                _,
            ) = await run_in_threadpool(run_calc)
        except Exception as e_calc:
            logging.error(f"Error in calculate_portfolio_summary: {e_calc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Calculation Error: {str(e_calc)}")

        raw = (overall_summary_metrics, summary_df, holdings_dict, account_level_metrics)
        _lru_put(_RAW_CALC_CACHE, raw_key, raw)
        return raw


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
        account_cash_mode_map,
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
    
    # PERF FIX (BN-03): Adaptive cache TTL — 60s during market hours, 300s after hours.
    # The previous 5-second bucket caused near-constant recalculation, even though the
    # frontend only refetches every 60s (market open) or 5min (staleTime).
    # Cache invalidation on data changes is handled by db_mtime in the key.
    cache_ttl_seconds = 60 if is_market_open() else 300
    time_key = int(time.time() / cache_ttl_seconds)
    
    cache_key = (
        currency,
        accounts_key,
        db_path,
        db_mtime,
        time_key
    )
    
    cached = _lru_get(_PORTFOLIO_SUMMARY_CACHE, cache_key)
    if cached is not None:
        logging.info(f"Using cached portfolio summary for key: {cache_key[:3]}...")
        return _filter_closed_positions(cached, show_closed_positions)

    logging.info(f"Summary Cache Miss. Time key: {time_key}")

    # Heavy portfolio math — shared with /summary/headline via _RAW_CALC_CACHE so
    # it runs at most once per cache window.
    (
        overall_summary_metrics,
        summary_df,
        holdings_dict,
        account_level_metrics,
    ) = await _compute_raw_summary(
        currency=currency,
        include_accounts=include_accounts,
        data=data,
        account_interest_rates=account_interest_rates,
        interest_free_thresholds=interest_free_thresholds,
    )
    # Copy before we enrich with TWR/dividend fields so the shared raw cache
    # entry stays a clean calculate_portfolio_summary output.
    if overall_summary_metrics is not None:
        overall_summary_metrics = dict(overall_summary_metrics)

    # --- Closed-account gating ---
    # If every account in `include_accounts` has a closure date on or before today,
    # rate-of-return metrics (TWR / IRR / yields) are unreliable due to residual
    # dividends arriving on a near-zero capital base. Skip TWR computation and gate
    # the affected response fields to None. Absolute-dollar metrics stay populated.
    closure_dates_map: Dict[str, str] = {}
    if config_manager:
        closure_dates_map = config_manager.gui_config.get("account_closure_dates", {}) or {}
    closed_accounts_in_slice, all_selected_closed = compute_account_closure_state(
        include_accounts, closure_dates_map, date.today()
    )

    # --- Calculate Annualized TWR and include in metrics ---
    # FIX: Always use live computation via _get_historical_performance_cached.
    # Previously, this used pre-calculated snapshots from portfolio_snapshots table
    # for "ALL" and single-account views. However, those snapshots were computed with
    # calc_method="STANDARD" (old worker path), while the performance graph uses
    # "numba_chrono" (from config). This caused a TWR mismatch between the dashboard
    # card and the performance graph. The live computation uses the same engine as the
    # graph and is already cached in-memory, so it's fast after first load.
    annualized_twr = None
    cumulative_twr = None
    if not df.empty and not all_selected_closed:
        try:
            df_for_twr = df
            if include_accounts:
                df_for_twr = df[df["Account"].isin(include_accounts)]
            if not df_for_twr.empty:
                hist_min_date = df_for_twr["Date"].min().date()
                hist_end_date = date.today()
                hist_daily_df, _, _, hist_status = await _get_historical_performance_cached(
                    df=df,
                    manual_overrides_dict=manual_overrides,
                    user_symbol_map=user_symbol_map,
                    user_excluded_symbols=user_excluded_symbols,
                    account_currency_map=account_currency_map,
                    original_csv_file_path=db_path,
                    start_date=hist_min_date,
                    end_date=hist_end_date,
                    interval="1d",
                    benchmark_symbols_yf=[],
                    display_currency=currency,
                    include_accounts=include_accounts,
                    account_cash_mode_map=account_cash_mode_map,
                    db_mtime=db_mtime,
                )
                final_twr_factor = None
                if "|||TWR_FACTOR:" in hist_status:
                    twr_part = hist_status.split("|||TWR_FACTOR:")[1].strip()
                    if twr_part.upper() != "NAN":
                        try:
                            final_twr_factor = float(twr_part)
                        except ValueError:
                            final_twr_factor = None
                if final_twr_factor is not None and final_twr_factor > 0:
                    days = (hist_end_date - hist_min_date).days
                    cumulative_twr = (final_twr_factor - 1) * 100.0
                    if days > 0:
                        annualized_factor = final_twr_factor ** (365.25 / days)
                        annualized_twr = (annualized_factor - 1) * 100.0
                    logging.info(
                        f"Summary Metrics: Live TWR computed for {len(include_accounts) if include_accounts else 'ALL'} accounts: factor={final_twr_factor:.4f}"
                    )
        except Exception as e_live_twr:
            logging.warning(f"Live-TWR computation failed: {e_live_twr}")

    if overall_summary_metrics:
        overall_summary_metrics["annualized_twr"] = annualized_twr
        overall_summary_metrics["cumulative_twr"] = cumulative_twr

        if not all_selected_closed:
            # --- NEW: Calculate Historical Dividend Metrics ---
            # Similar to TWR, these are based on total history (since inception)
            total_dividends = overall_summary_metrics.get("dividends", 0.0)
            total_buy_cost = overall_summary_metrics.get("total_buy_cost", 0.0)

            # We need 'days' since inception for the annualized calculation.
            # Recalculate 'days' if not already in scope from one of the TWR branches.
            days_since_inception = 0
            try:
                df_for_days = df
                if include_accounts:
                    df_for_days = df[df["Account"].isin(include_accounts)]
                if not df_for_days.empty:
                    min_date_val = df_for_days["Date"].min().date()
                    days_since_inception = (date.today() - min_date_val).days
            except Exception:
                pass

            if abs(total_buy_cost) > 1e-9:
                # 1. Cumulative Historical Dividend Return % (Yield on Cost)
                div_cum = (total_dividends / total_buy_cost) * 100.0
                overall_summary_metrics["dividend_return_cumulative"] = div_cum

                # 2. Annualized Historical Dividend Return %
                if days_since_inception > 0:
                    # Geometric annualization: (1 + total_div/total_cost)^(365.25/days) - 1
                    div_factor = 1.0 + (total_dividends / total_buy_cost)
                    if div_factor > 0:
                        annual_div_factor = div_factor ** (365.25 / days_since_inception)
                        overall_summary_metrics["dividend_return_annualized"] = (annual_div_factor - 1) * 100.0
            # --- END NEW ---

        # Gate all rate-of-return metrics for closed-account slices. Absolute-dollar
        # metrics (cash balance, realized gain, dividends, fees, taxes, ...) keep
        # their historical values.
        if all_selected_closed:
            for key in (
                "annualized_twr",
                "cumulative_twr",
                "portfolio_mwr",
                "ytd_return",
                "dividend_return_cumulative",
                "dividend_return_annualized",
                "total_return_pct",
            ):
                overall_summary_metrics[key] = None

        overall_summary_metrics["all_selected_closed"] = all_selected_closed
        overall_summary_metrics["closed_accounts"] = closed_accounts_in_slice

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
    
    _lru_put(_PORTFOLIO_SUMMARY_CACHE, cache_key, result)

    return _filter_closed_positions(result, show_closed_positions)


async def _calculate_historical_performance_internal(
    currency: str,
    period: str,
    accounts: Optional[List[str]],
    benchmarks: List[str],
    data: tuple,
    return_df: bool = False,
    interval: str = "1d",
    from_date_str: Optional[str] = None,
    to_date_str: Optional[str] = None,
    force: bool = False,
    end_date_cap: Optional[date] = None,
):
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
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
    from utils_time import is_tradable_day, get_nyse_calendar
    
    # UNIFIED: Use get_est_today() + 1 day as the exclusive end_date for all relative periods.
    # This ensures yfinance includes today's data (if available) and the graph isn't truncated.
    if to_date_custom:
        end_date = to_date_custom
    else:
        end_date = get_est_today() + timedelta(days=1)

    # When every selected account is closed, the caller passes end_date_cap so the
    # graph ends at the closure date instead of trailing flat to today.
    if end_date_cap is not None:
        cap_exclusive = end_date_cap + timedelta(days=1)
        if end_date > cap_exclusive:
            end_date = cap_exclusive
        if to_date_custom and to_date_custom > cap_exclusive:
            to_date_custom = cap_exclusive
    
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
        daily_df, _, historical_fx_yf, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=start_date,
            end_date=end_date,
            display_currency=currency,
            include_accounts=accounts,
            benchmark_symbols_yf=benchmarks,
            interval=calc_interval,
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=data[7]  # db_mtime
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
             # Standardized Baseline Search
             def get_robust_divisor(series):
                 if series.empty:
                     return 1.0
                 # Try first point
                 v0 = series.iloc[0]
                 if pd.notna(v0) and v0 != 0:
                     return v0
                 # Search forward for first non-zero/non-nan
                 valid_points = series[series.notna() & (series != 0)]
                 if not valid_points.empty:
                     return valid_points.iloc[0]
                 return 1.0

             # 1. Normalize Portfolio TWR
             if "Portfolio Accumulated Gain" in daily_df.columns:
                 start_val = get_robust_divisor(daily_df["Portfolio Accumulated Gain"])
                 if start_val != 1.0 or daily_df["Portfolio Accumulated Gain"].iloc[0] == 0:
                      daily_df["Portfolio Accumulated Gain"] = daily_df["Portfolio Accumulated Gain"] / start_val

             # 2. Normalize Benchmarks
             if benchmarks:
                 for b_ticker in benchmarks:
                     bm_col = f"{b_ticker} Accumulated Gain"
                     if bm_col in daily_df.columns:
                         bm_start_val = get_robust_divisor(daily_df[bm_col])
                         if bm_start_val != 1.0 or daily_df[bm_col].iloc[0] == 0:
                             daily_df[bm_col] = daily_df[bm_col] / bm_start_val
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

    result = []
    # daily_df index is Date

    ticker_to_name = {v: k for k, v in config.BENCHMARK_MAPPING.items()}
    
    # --- FX Rate series preparation ---
    fx_rate_series = None
    if currency and currency.upper() != "USD" and historical_fx_yf:
        curr_upper = currency.upper()
        if curr_upper == "THB":
            fx_pair = "USDTHB=X"
        else:
            fx_pair = f"{curr_upper}=X"
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
                     if pre_market_count == 0:
                         logging.info(f"First Pre-Market Reject: {dt_ny} (from {dt})")
                     pre_market_count += 1
                 else:
                     post_market_count += 1
                 continue
             else:
                 if filtered_count == 0:
                     logging.info(f"First Accepted: {dt_ny} (from {dt})")

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
