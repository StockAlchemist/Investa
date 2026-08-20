"""Portfolio routes: summary, holdings, history, asset change, health, AI review."""

# ruff: noqa: E402
import asyncio
import logging
import sqlite3
import time
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from db_utils import get_cached_screener_results
from finutils import is_cash_symbol
from market_data import map_to_yf_symbol
from portfolio_analyzer import (
    calculate_periodic_returns,
    calculate_fifo_lots_and_gains,
)
from risk_metrics import calculate_all_risk_metrics
from server.auth import User
from server.dependencies import (
    get_config_manager,
    get_current_user,
    get_transaction_data,
    get_user_db_connection,
)
from server.portfolio_service import (
    _PORTFOLIO_SUMMARY_CACHE,
    _calculate_historical_performance_internal,
    _calculate_portfolio_summary_internal,
    _compute_raw_summary,
    _get_historical_performance_cached,
    compute_account_closure_state,
)
from server.route_utils import _lru_get, clean_nans, get_mdp
from utils_time import is_market_open

router = APIRouter()


@router.get("/asset_change")
async def get_asset_change(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
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
        account_cash_mode_map,  # NEW
        original_csv_path,
        _,  # Ignore db_mtime
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
        daily_df, _, _, final_status_str = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=date(2000, 1, 1),  # All history
            end_date=date.today(),
            display_currency=currency,
            include_accounts=accounts,
            benchmark_symbols_yf=mapped_benchmarks,
            interval="D",
            account_cash_mode_map=account_cash_mode_map,  # PASSING IT HERE
            db_mtime=data[7],  # db_mtime
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
                    ticker_part = col[: -len(suffix)]
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
                    p_df.index.name = "Date"

                p_df_reset = p_df.reset_index()

                # Convert dates to strings and handle 'index' fallback if needed
                if "Date" in p_df_reset.columns:
                    p_df_reset["Date"] = p_df_reset["Date"].astype(str)
                elif "index" in p_df_reset.columns:
                    # Fallback rename if for some reason it's still called 'index'
                    p_df_reset.rename(columns={"index": "Date"}, inplace=True)
                    p_df_reset["Date"] = p_df_reset["Date"].astype(str)

                result[period] = clean_nans(p_df_reset.to_dict(orient="records"))
            else:
                result[period] = []

        return result

    except Exception as e:
        logging.error(f"Error calculating asset change: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to calculate asset change")


@router.get("/summary")
async def get_portfolio_summary(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    show_closed: Optional[bool] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
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
        account_cash_mode_map,
        original_csv_path,
        _,
    ) = data

    if df.empty:
        return {"error": "No transaction data available"}

    try:
        # Calculate portfolio summary using helper
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=show_closed if show_closed is not None else True,
            data=data,
            current_user=current_user,
        )

        overall_summary_metrics = summary_data["metrics"]
        account_level_metrics = summary_data["account_metrics"]

        # --- Fetch Market Indices ---
        if overall_summary_metrics:
            try:
                # cache_only: never block the summary response on a live index
                # fetch (~7-8s). The dedicated /indices endpoint keeps this warm
                # and the frontend renders the header from its own query.
                mdp = get_mdp()
                indices_data = mdp.get_index_quotes(
                    config.INDICES_FOR_HEADER, cache_only=True
                )
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
            safe_holdings_dict = {
                f"{sym}|{acc}": val for (sym, acc), val in holdings_dict_raw.items()
            }

        response_data = {
            "metrics": overall_summary_metrics,
            "account_metrics": account_level_metrics,
            "summary_df": serialized_df,
            "holdings_dict": safe_holdings_dict,
        }
        return clean_nans(response_data)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error calculating summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to calculate portfolio summary"
        )


@router.get("/summary/headline")
async def get_portfolio_summary_headline(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Fast path for the top card: total value, day change, and the other headline
    metrics — and nothing else.

    Shares the heavy calculation cache with /summary but SKIPS the expensive
    historical TWR/dividend step, the index fetch, and the holdings/summary_df
    serialization. This lets the dashboard's headline card render and update as
    soon as the core math finishes, well before the full dashboard is ready.
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

    if df.empty:
        return {"metrics": {}}

    # If the full summary is already cached, it's a superset — reuse it.
    accounts_key = tuple(sorted(accounts)) if accounts else "ALL"
    cache_ttl_seconds = 60 if is_market_open() else 300
    time_key = int(time.time() / cache_ttl_seconds)
    full_key = (currency, accounts_key, db_path, db_mtime, time_key)
    cached_full = _lru_get(_PORTFOLIO_SUMMARY_CACHE, full_key)
    if cached_full is not None and cached_full.get("metrics"):
        return clean_nans({"metrics": cached_full["metrics"]})

    # Interest settings are inputs to the calculation.
    account_interest_rates: dict = {}
    interest_free_thresholds: dict = {}
    config_manager = get_config_manager(current_user) if current_user else None
    if config_manager:
        config_manager.load_manual_overrides()
        account_interest_rates = config_manager.manual_overrides.get(
            "account_interest_rates", {}
        )
        interest_free_thresholds = config_manager.manual_overrides.get(
            "interest_free_thresholds", {}
        )

    try:
        overall_summary_metrics, _, _, _ = await _compute_raw_summary(
            currency=currency,
            include_accounts=accounts,
            data=data,
            account_interest_rates=account_interest_rates,
            interest_free_thresholds=interest_free_thresholds,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error calculating headline summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to calculate portfolio summary headline"
        )

    metrics = dict(overall_summary_metrics) if overall_summary_metrics else {}

    # Cheap closure-state gating (date math only, no network) so the card matches
    # /summary for closed-account slices. Rate-of-return fields aren't computed on
    # this path, so they're simply absent/None — the card doesn't need them.
    closure_dates_map: Dict[str, str] = {}
    if config_manager:
        closure_dates_map = (
            config_manager.gui_config.get("account_closure_dates", {}) or {}
        )
    closed_in_slice, all_selected_closed = compute_account_closure_state(
        accounts, closure_dates_map, date.today()
    )
    metrics["all_selected_closed"] = all_selected_closed
    metrics["closed_accounts"] = closed_in_slice
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
            metrics[key] = None

    return clean_nans({"metrics": metrics})


@router.post("/portfolio/ai_review")
async def get_portfolio_ai_review(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    refresh: bool = False,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """
    Generates or retrieves a cached AI review for the portfolio.
    """
    from server.portfolio_ai_analyzer import generate_portfolio_review

    (df, manual, user_map, excluded, acc_curr, cash_mode, path, mtime) = data

    if df.empty:
        raise HTTPException(status_code=400, detail="Portfolio is empty.")

    try:
        # 1. Get Summary
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            data=data,
            current_user=current_user,
        )

        # 2. Get Risk Metrics
        # We need historical data for risk metrics
        # Use existing cache helper for history
        min_date = df["Date"].min().date()
        daily_df, _, _, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual,
            user_symbol_map=user_map,
            user_excluded_symbols=excluded,
            account_currency_map=acc_curr,
            original_csv_file_path=path,
            start_date=min_date,  # Full history for better risk stats
            end_date=date.today(),
            interval="D",
            benchmark_symbols_yf=["SPY"],  # Benchmark against SPY for Beta
            display_currency=currency,
            include_accounts=accounts,
            account_cash_mode_map=cash_mode,
            db_mtime=mtime,
        )

        # Calculate risk metrics
        # We need historical data for this. Using a default period of 1y for risk analysis

        risk_metrics = {}
        try:
            # Unpack data dependency
            (
                df,
                manual_overrides,
                user_symbol_map,
                user_excluded_symbols,
                account_currency_map,
                account_cash_mode_map,
                original_csv_path,
                mtime,
            ) = data

            start_date = date.today() - timedelta(days=365)
            end_date = date.today()

            daily_df, _, _, _ = await _get_historical_performance_cached(
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
                benchmark_symbols_yf=["^GSPC"],  # Fetch S&P 500 for Beta/Alpha
                interval="D",
                account_cash_mode_map=account_cash_mode_map,
                db_mtime=mtime,
            )

            if daily_df is not None and "Portfolio Value" in daily_df.columns:
                portfolio_values = daily_df["Portfolio Value"]
                benchmark_values = (
                    daily_df["^GSPC Price"]
                    if "^GSPC Price" in daily_df.columns
                    else None
                )
                risk_metrics = clean_nans(
                    calculate_all_risk_metrics(
                        portfolio_values, benchmark_values=benchmark_values
                    )
                )

            # Fallback if empty - Just log it, don't use mock data for production
            if not risk_metrics:
                # Initialize with N/A to ensure UI handles it gracefully without crashing
                risk_metrics = {
                    "sharpe_ratio": "N/A",
                    "sortino_ratio": "N/A",
                    "volatility": "N/A",
                    "max_drawdown": "N/A",
                    "beta": "N/A",
                    "alpha": "N/A",
                }

        except Exception as e:
            logging.error(f"AI Review Risk Metrics Error: {e}", exc_info=True)

        # Prepare holdings list from summary_df (which has rich data like Sector, Country)
        holdings_list = []
        if "summary_df" in summary_data and isinstance(
            summary_data["summary_df"], pd.DataFrame
        ):
            sdf = summary_data["summary_df"]
            if not sdf.empty:
                # Filter for active holdings only
                if "Quantity" in sdf.columns:
                    sdf = sdf[abs(sdf["Quantity"]) > 1e-6].copy()

                # Normalize keys for the analyzer
                # The columns might be "Market Value (USD)", "Symbol", "Sector", etc.
                # We rename them to standard keys
                rename_map = {
                    "Symbol": "symbol",
                    "Sector": "sector",
                    "Country": "country",
                    "quoteType": "asset_type",
                    # Dynamic columns handled below
                }

                # Handle dynamic currency columns
                mv_col = [c for c in sdf.columns if c.startswith("Market Value (")]
                if mv_col:
                    rename_map[mv_col[0]] = "market_value"

                gain_col = [c for c in sdf.columns if c.startswith("Unrealized Gain (")]
                if gain_col:
                    rename_map[gain_col[0]] = "unrealized_gain"

                alloc_col = [c for c in sdf.columns if "% Portfolio" in c]
                if alloc_col:
                    rename_map[alloc_col[0]] = "allocation_percent"

                # Convert
                records = sdf.to_dict(orient="records")
                for r in records:
                    new_r = {}
                    for k, v in r.items():
                        # Map known keys
                        if k in rename_map:
                            new_r[rename_map[k]] = v
                        # Keep others as-is (lowercased)
                        else:
                            new_r[k.lower().replace(" ", "_")] = v
                    holdings_list.append(new_r)

        # Fallback to holdings_dict if summary_df processing failed or was empty
        if not holdings_list and "holdings_dict" in summary_data:
            print(
                "DEBUG: AI Review - Fallback to holdings_dict (summary_df empty/missing)"
            )
            holdings_list = list(summary_data["holdings_dict"].values())

        # Inject holdings list into portfolio data for analyzer
        summary_data["holdings"] = holdings_list

        # Generate review — pass the user DB so screener_cache lookups read the
        # same rows the rest of the app sees, and so the cache hash fingerprint
        # invalidates when those rows refresh mid-day.
        review = generate_portfolio_review(
            portfolio_data=summary_data,
            risk_metrics=risk_metrics,
            force_refresh=refresh,
            db_conn=db_conn,
        )

        return review

    except Exception as e:
        logging.error(f"Portfolio AI Review Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to generate portfolio review"
        )


@router.get("/holdings")
async def get_holdings(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    show_closed: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection),
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
        account_cash_mode_map,
        original_csv_path,
        _,
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
            show_closed_positions=show_closed,  # Pass the parameter
            data=data,
            current_user=current_user,
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
                            if (
                                str(h_key[0]).lower() == str(sym).lower()
                                and str(h_key[1]).lower() == str(acct).lower()
                            ):
                                record["lots"] = h_data.get("lots", [])
                                break
        # ----------------------------------------------------

        # --- ADDED: Include AI Score and Intrinsic Value from Screener Cache ---
        try:
            # Get unique symbols
            symbols = list(set(r.get("Symbol") for r in records if r.get("Symbol")))
            logging.info(
                f"[DEBUG_HOLDINGS] Fetching screener data for {len(symbols)} symbols: {symbols[:10]}..."
            )
            if symbols:
                screener_data = get_cached_screener_results(symbols)
                logging.info(
                    f"[DEBUG_HOLDINGS] Found screener data for {len(screener_data)} / {len(symbols)} symbols."
                )

                # Merge into holdings records
                match_count = 0
                for record in records:
                    sym = record.get("Symbol")
                    if sym:
                        u_sym = sym.upper()
                        if u_sym in screener_data:
                            s_info = screener_data[u_sym]
                            record["ai_score"] = s_info.get("ai_score")
                            iv_local = s_info.get("intrinsic_value")
                            fx = record.get("fx_rate", 1.0)
                            if iv_local is not None and fx is not None:
                                converted_iv = iv_local * fx
                                record["intrinsic_value"] = converted_iv

                                # Recalculate Margin of Safety using Display Price
                                price_display = record.get(f"Price ({currency})")
                                if (
                                    pd.notna(price_display)
                                    and price_display is not None
                                    and converted_iv > 1e-9
                                ):
                                    record["margin_of_safety"] = (
                                        (converted_iv - price_display)
                                        / converted_iv
                                        * 100
                                    )
                                else:
                                    record["margin_of_safety"] = s_info.get(
                                        "margin_of_safety"
                                    )
                            else:
                                record["intrinsic_value"] = iv_local
                                record["margin_of_safety"] = s_info.get(
                                    "margin_of_safety"
                                )

                            record["has_ai_review"] = s_info.get("has_ai_review")
                            record["ai_sentiment"] = s_info.get("ai_sentiment")
                            record["ai_catalysts"] = s_info.get("ai_catalysts")
                            match_count += 1
                logging.info(
                    f"[DEBUG_HOLDINGS] Successfully merged data for {match_count} records."
                )
        except Exception as e_ai:
            logging.warning(f"Error merging AI data into holdings: {e_ai}")
        # ----------------------------------------------------

        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting holdings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch holdings")


# Period windows (in calendar days) for the holdings performance heatmap.
# "ytd" is handled separately (anchored to Jan 1 of the current year).
_HEATMAP_PERIOD_DAYS = {"1m": 30, "3m": 91, "6m": 182, "1y": 365}


def _period_pct_return(price: pd.Series, start: date) -> Optional[float]:
    """Price-only percent return between the close on/before ``start`` and the
    latest close. Returns None when there's no usable data. Falls back to the
    earliest available price when the series doesn't reach back to ``start``
    (e.g. a recently opened position), so newer holdings still render."""
    if price is None or price.empty:
        return None
    s = price.dropna()
    if s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    start_ts = pd.Timestamp(start)
    if getattr(s.index, "tz", None) is not None:
        start_ts = start_ts.tz_localize(s.index.tz)

    last = s.iloc[-1]
    prior = s.loc[s.index <= start_ts]
    base = prior.iloc[-1] if not prior.empty else s.iloc[0]
    if pd.isna(base) or pd.isna(last) or base == 0:
        return None
    return (last / base - 1.0) * 100.0


@router.get("/holdings/returns")
async def get_holdings_returns(
    symbols: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """Period price returns (%) per holding symbol for the performance heatmap.

    Returns ``{symbol: {"1m","3m","6m","1y","ytd"}}`` percent price returns
    (dividends excluded, matching a Finviz-style performance map). The client
    passes its current holding symbols via ``symbols``; if omitted, every
    non-cash symbol in the transaction history is used.
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, *_ = data
    if df.empty:
        return {}

    universe = symbols if symbols else df["Symbol"].dropna().unique().tolist()
    # Map each user-facing symbol to its yfinance ticker, skipping cash and
    # anything unmappable/excluded.
    yf_map: Dict[str, str] = {}
    for sym in universe:
        if not sym or is_cash_symbol(sym):
            continue
        yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols)
        if yf_sym:
            yf_map[sym] = yf_sym
    if not yf_map:
        return {}

    today = date.today()
    period_starts: Dict[str, date] = {
        label: today - timedelta(days=days)
        for label, days in _HEATMAP_PERIOD_DAYS.items()
    }
    period_starts["ytd"] = date(today.year, 1, 1)
    # One fetch covering the longest window (+buffer for the as-of lookup).
    fetch_start = min(period_starts.values()) - timedelta(days=7)

    provider = get_mdp()
    yf_symbols = sorted(set(yf_map.values()))
    try:
        # get_historical_data is sync (DB + possible network); keep the event loop free.
        hist, _ = await asyncio.to_thread(
            provider.get_historical_data, yf_symbols, fetch_start, today
        )
    except Exception as e:
        logging.error(f"holdings/returns: historical fetch failed: {e}")
        return {}

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for sym, yf_sym in yf_map.items():
        df_sym = hist.get(yf_sym)
        if df_sym is None or df_sym.empty or "price" not in df_sym.columns:
            continue
        price = df_sym["price"]
        result[sym] = {
            label: _period_pct_return(price, start)
            for label, start in period_starts.items()
        }

    return clean_nans(result)


@router.get("/stock/{symbol}/position")
async def get_stock_position(
    symbol: str,
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """Calculates comprehensive position summary, FIFO lots, and return attribution for a single stock."""
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
        original_csv_path,
        db_mtime,
    ) = data

    sym_clean = symbol.upper().strip()
    yf_mapped = map_to_yf_symbol(sym_clean, user_symbol_map, user_excluded_symbols)
    sym_variants = {sym_clean}
    if yf_mapped:
        sym_variants.add(yf_mapped.upper().strip())

    if df.empty:
        return {
            "symbol": sym_clean,
            "display_currency": currency,
            "local_currency": currency,
            "fx_rate": 1.0,
            "has_position": False,
            "summary": None,
            "returns": None,
            "open_lots": [],
            "closed_trades": [],
        }

    # Check if this symbol has transactions
    sym_series = df["Symbol"].fillna("").astype(str).str.upper().str.strip()
    sym_mask = sym_series.isin(sym_variants)
    if "To Account" in df.columns:
        # Include transfer rows where symbol matches
        sym_mask = sym_mask | sym_series.isin(sym_variants)
    df_sym = df[sym_mask]

    if df_sym.empty:
        return {
            "symbol": sym_clean,
            "display_currency": currency,
            "local_currency": currency,
            "fx_rate": 1.0,
            "has_position": False,
            "summary": None,
            "returns": None,
            "open_lots": [],
            "closed_trades": [],
        }

    try:
        # 1. Historical FX for currency conversions
        min_date = df["Date"].min().date()
        max_date = date.today()
        _, _, historical_fx_yf, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=min_date,
            end_date=max_date,
            interval="D",
            benchmark_symbols_yf=[],
            display_currency=currency,
            include_accounts=accounts,
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=db_mtime,
        )

        # 2. FIFO Lots & Realized Gains
        tx_to_process = df.copy()
        tx_to_process.sort_values(by=["Date", "original_index"], inplace=True)
        df_gains, open_lots_dict = calculate_fifo_lots_and_gains(
            transactions_df=tx_to_process,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            shortable_symbols=config.SHORTABLE_SYMBOLS,
            stock_quantity_close_tolerance=config.STOCK_QUANTITY_CLOSE_TOLERANCE,
        )

        # 3. Get exact holding metrics via summary generator
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=True,
            data=data,
            current_user=current_user,
        )

        overall_metrics = summary_data.get("metrics", {}) or {}
        overall_mkt_val = overall_metrics.get("market_value", 0.0) or 0.0

        summary_df = summary_data.get("summary_df")
        all_rows = []
        if summary_df is not None and not summary_df.empty:
            summary_df_clean = summary_df.where(pd.notnull(summary_df), None)
            all_rows = summary_df_clean.to_dict(orient="records")

        matching_rows = [
            r
            for r in all_rows
            if str(r.get("Symbol", "")).upper().strip() in sym_variants
        ]

        # Extract price and FX info
        current_price = 0.0
        local_currency = currency
        fx_rate = 1.0

        if matching_rows:
            current_price = float(
                matching_rows[0].get(
                    f"Price ({currency})", matching_rows[0].get("Price", 0.0)
                )
                or 0.0
            )
            local_currency = matching_rows[0].get("Local Currency", currency)
            fx_rate = float(matching_rows[0].get("fx_rate", 1.0) or 1.0)
        else:
            # Try to get price from MarketDataProvider
            mdp = get_mdp()
            quotes, _, _, _, _ = mdp.get_current_quotes(
                [sym_clean], {currency}, {}, set()
            )
            stock_data = quotes.get(sym_clean, {})
            current_price = float(stock_data.get("price", 0.0) or 0.0)

        # 4. Build Open Lots list
        open_lots_list = []
        accounts_filter_norm = (
            {str(a).upper().strip() for a in accounts} if accounts else None
        )

        for (s, acc), lots in open_lots_dict.items():
            if str(s).upper().strip() not in sym_variants:
                continue
            if accounts_filter_norm and acc.upper().strip() not in accounts_filter_norm:
                continue

            for idx, lot in enumerate(lots):
                qty = float(lot.get("qty", 0.0))
                if qty > config.STOCK_QUANTITY_CLOSE_TOLERANCE:
                    purchase_date = lot.get("purchase_date")
                    cps_local = float(lot.get("cost_per_share_local_net", 0.0))
                    lot_fx = float(
                        lot.get("purchase_fx_to_display", fx_rate) or fx_rate
                    )
                    lot_cost_basis_display = qty * cps_local * lot_fx
                    lot_mkt_val_display = qty * current_price
                    lot_unreal_gain = lot_mkt_val_display - lot_cost_basis_display
                    lot_unreal_gain_pct = (
                        (lot_unreal_gain / lot_cost_basis_display * 100.0)
                        if lot_cost_basis_display > 1e-6
                        else 0.0
                    )
                    days_held = (
                        (date.today() - purchase_date).days if purchase_date else 0
                    )
                    term = "long_term" if days_held >= 365 else "short_term"

                    open_lots_list.append(
                        {
                            "lot_id": int(lot.get("original_tx_id") or (idx + 1)),
                            "date": str(purchase_date),
                            "account": acc,
                            "quantity": round(qty, 6),
                            "cost_per_share_local": round(cps_local, 4),
                            "cost_basis_display": round(lot_cost_basis_display, 2),
                            "market_value_display": round(lot_mkt_val_display, 2),
                            "unrealized_gain_display": round(lot_unreal_gain, 2),
                            "unrealized_gain_pct": round(lot_unreal_gain_pct, 2),
                            "holding_period_days": days_held,
                            "tax_term": term,
                        }
                    )

        # Sort lots chronologically (FIFO order)
        open_lots_list.sort(key=lambda x: (x["date"], x["lot_id"]))

        # 5. Build Closed Trades list
        closed_trades_list = []
        if not df_gains.empty:
            df_gains_sym = df_gains[
                df_gains["Symbol"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .isin(sym_variants)
            ]
            if accounts_filter_norm:
                df_gains_sym = df_gains_sym[
                    df_gains_sym["Account"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .isin(accounts_filter_norm)
                ]

            for _, row in df_gains_sym.iterrows():
                closed_trades_list.append(
                    {
                        "sell_date": str(row.get("Date")),
                        "account": str(row.get("Account")),
                        "quantity_sold": float(row.get("Quantity", 0.0)),
                        "sale_price": float(row.get("Avg Sale Price (Local)", 0.0)),
                        "proceeds_display": float(
                            row.get("Total Proceeds (Display)", 0.0)
                        ),
                        "cost_basis_display": float(
                            row.get("Total Cost Basis (Display)", 0.0)
                        ),
                        "realized_gain_display": float(
                            row.get("Realized Gain (Display)", 0.0)
                        ),
                        "original_tx_id": int(row.get("original_tx_id") or 0),
                    }
                )

        closed_trades_list.sort(key=lambda x: x["sell_date"], reverse=True)

        # 6. Build Aggregated Position Summary & Return Attribution
        tot_qty = sum(float(r.get("Quantity", 0.0) or 0.0) for r in matching_rows)
        tot_mkt_val = sum(
            float(
                r.get(f"Market Value ({currency})", r.get("Market Value", 0.0)) or 0.0
            )
            for r in matching_rows
        )
        tot_cost_basis = sum(
            float(r.get(f"Cost Basis ({currency})", r.get("Cost Basis", 0.0)) or 0.0)
            for r in matching_rows
        )
        tot_buy_cost = sum(
            float(
                r.get(f"Total Buy Cost ({currency})", r.get("Total Buy Cost", 0.0))
                or 0.0
            )
            for r in matching_rows
        )
        tot_unreal_gain = sum(
            float(
                r.get(f"Unreal. Gain ({currency})", r.get("Unreal. Gain", 0.0)) or 0.0
            )
            for r in matching_rows
        )
        tot_real_gain = sum(
            float(
                r.get(f"Realized Gain ({currency})", r.get("Realized Gain", 0.0)) or 0.0
            )
            for r in matching_rows
        )
        tot_divs = sum(
            float(r.get(f"Dividends ({currency})", r.get("Dividends", 0.0)) or 0.0)
            for r in matching_rows
        )
        tot_comm = sum(
            float(r.get(f"Commissions ({currency})", r.get("Commissions", 0.0)) or 0.0)
            for r in matching_rows
        )
        tot_tax = sum(
            float(r.get(f"Taxes ({currency})", r.get("Taxes", 0.0)) or 0.0)
            for r in matching_rows
        )
        tot_gain = sum(
            float(r.get(f"Total Gain ({currency})", r.get("Total Gain", 0.0)) or 0.0)
            for r in matching_rows
        )
        tot_fx_gain = sum(
            float(r.get(f"FX Gain/Loss ({currency})", 0.0) or 0.0)
            for r in matching_rows
            if pd.notna(r.get(f"FX Gain/Loss ({currency})"))
        )

        # For open positions, FIFO cost basis and unrealized gain strictly come from the unliquidated open lots
        if open_lots_list:
            lot_qty = sum(lot["quantity"] for lot in open_lots_list)
            lot_cost = sum(lot["cost_basis_display"] for lot in open_lots_list)
            if lot_qty > 1e-6:
                tot_qty = lot_qty
            if lot_cost > 0:
                tot_cost_basis = lot_cost
                tot_unreal_gain = tot_mkt_val - tot_cost_basis
                if tot_buy_cost <= 1e-6:
                    tot_buy_cost = tot_cost_basis
                tot_gain = (
                    tot_unreal_gain + tot_real_gain + tot_divs - tot_comm - tot_tax
                )
        elif abs(tot_qty) < 1e-6 and not open_lots_list:
            tot_cost_basis = 0.0
            tot_unreal_gain = 0.0

        avg_cost_price = (tot_cost_basis / tot_qty) if abs(tot_qty) > 1e-6 else 0.0
        unreal_gain_pct = (
            (tot_unreal_gain / tot_cost_basis * 100.0)
            if abs(tot_cost_basis) > 1e-6
            else 0.0
        )
        total_return_pct = (
            (tot_gain / tot_buy_cost * 100.0) if abs(tot_buy_cost) > 1e-6 else 0.0
        )
        fx_gain_pct = (
            (tot_fx_gain / tot_cost_basis * 100.0)
            if abs(tot_cost_basis) > 1e-6
            else 0.0
        )
        port_weight_pct = (
            (tot_mkt_val / overall_mkt_val * 100.0) if overall_mkt_val > 1e-6 else 0.0
        )

        irr_val = matching_rows[0].get("IRR (%)") if matching_rows else None
        yield_cost = (
            matching_rows[0].get("Div. Yield (Cost) %") if matching_rows else None
        )
        yield_mkt = (
            matching_rows[0].get("Div. Yield (Current) %") if matching_rows else None
        )
        iad = (
            matching_rows[0].get("Indicated Annual Dividend", 0.0)
            if matching_rows
            else 0.0
        )

        summary_obj = {
            "quantity": round(tot_qty, 6),
            "current_price": round(current_price, 4),
            "market_value": round(tot_mkt_val, 2),
            "avg_cost_price": round(avg_cost_price, 4),
            "cost_basis": round(tot_cost_basis, 2),
            "total_buy_cost": round(tot_buy_cost, 2),
            "portfolio_weight_pct": round(port_weight_pct, 2)
            if port_weight_pct
            else None,
        }

        returns_obj = {
            "unrealized_gain": round(tot_unreal_gain, 2),
            "unrealized_gain_pct": round(unreal_gain_pct, 2),
            "realized_gain": round(tot_real_gain, 2),
            "lifetime_dividends": round(tot_divs, 2),
            "commissions": round(tot_comm, 2),
            "withholding_taxes": round(tot_tax, 2),
            "total_gain": round(tot_gain, 2),
            "total_return_pct": round(total_return_pct, 2),
            "irr_pct": round(float(irr_val), 2)
            if irr_val is not None and pd.notna(irr_val)
            else None,
            "twrr_pct": None,
            "indicated_annual_dividend": round(float(iad), 4) if iad else 0.0,
            "yield_on_cost_pct": round(float(yield_cost), 2)
            if yield_cost is not None and pd.notna(yield_cost)
            else None,
            "market_yield_pct": round(float(yield_mkt), 2)
            if yield_mkt is not None and pd.notna(yield_mkt)
            else None,
            "fx_gain_loss": round(tot_fx_gain, 2),
            "fx_gain_loss_pct": round(fx_gain_pct, 2),
        }

        has_pos = (
            (abs(tot_qty) > 1e-6)
            or len(closed_trades_list) > 0
            or len(open_lots_list) > 0
        )

        res = {
            "symbol": sym_clean,
            "display_currency": currency,
            "local_currency": local_currency,
            "fx_rate": fx_rate,
            "has_position": has_pos,
            "summary": summary_obj,
            "returns": returns_obj,
            "open_lots": open_lots_list,
            "closed_trades": closed_trades_list,
        }

        return clean_nans(res)

    except Exception as e:
        logging.error(f"Error getting stock position for {symbol}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get position data for {symbol}"
        )


@router.get("/stock/{symbol}/position_history")
async def get_stock_position_history(
    symbol: str,
    currency: str = "USD",
    period: str = "1y",
    benchmarks: Optional[List[str]] = Query(None),
    accounts: Optional[List[str]] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Returns historical daily position performance (Market Value, Cost Basis, Unrealized Gain, Return %)
    for a single stock position, with optional benchmark comparisons.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        account_cash_mode_map,
        original_csv_path,
        db_mtime,
    ) = data

    sym_clean = symbol.upper().strip()
    yf_mapped = map_to_yf_symbol(sym_clean, user_symbol_map, user_excluded_symbols)
    sym_variants = {sym_clean}
    if yf_mapped:
        sym_variants.add(yf_mapped.upper().strip())

    if df.empty:
        return []

    # Filter transactions for this symbol
    sym_series = df["Symbol"].fillna("").astype(str).str.upper().str.strip()
    sym_mask = sym_series.isin(sym_variants)
    df_sym = df[sym_mask].copy()

    accounts_filter_norm = (
        {str(a).upper().strip() for a in accounts}
        if isinstance(accounts, (list, set, tuple)) and accounts
        else None
    )
    if accounts_filter_norm and not df_sym.empty:
        df_sym = df_sym[
            df_sym["Account"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .isin(accounts_filter_norm)
        ]

    if df_sym.empty:
        return []

    try:
        from utils_time import get_est_today

        today = get_est_today()
        end_date = today + timedelta(days=1)

        first_tx_date = df_sym["Date"].min().date()

        from_date_custom = None
        to_date_custom = None
        if from_date and isinstance(from_date, str):
            try:
                from_date_custom = datetime.strptime(from_date, "%Y-%m-%d").date()
            except Exception:
                pass
        if to_date and isinstance(to_date, str):
            try:
                to_date_custom = datetime.strptime(to_date, "%Y-%m-%d").date()
                end_date = to_date_custom + timedelta(days=1)
            except Exception:
                pass

        if period == "custom" and from_date_custom:
            display_start_date = from_date_custom
        elif period == "1m":
            display_start_date = end_date - timedelta(days=30)
        elif period == "3m":
            display_start_date = end_date - timedelta(days=90)
        elif period == "6m":
            display_start_date = end_date - timedelta(days=180)
        elif period == "ytd":
            display_start_date = date(today.year, 1, 1)
        elif period == "1y":
            display_start_date = end_date - timedelta(days=365)
        elif period == "3y":
            display_start_date = end_date - timedelta(days=365 * 3)
        elif period == "5y":
            display_start_date = end_date - timedelta(days=365 * 5)
        elif period == "all" or period == "max":
            display_start_date = first_tx_date
        else:
            display_start_date = end_date - timedelta(days=365)

        calc_start_date = min(first_tx_date, display_start_date)

        # 1. Historical FX for currency conversions
        _, _, historical_fx_yf, _ = await _get_historical_performance_cached(
            df=df,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            account_currency_map=account_currency_map,
            original_csv_file_path=original_csv_path,
            start_date=calc_start_date,
            end_date=end_date,
            interval="D",
            benchmark_symbols_yf=[],
            display_currency=currency,
            include_accounts=accounts if isinstance(accounts, list) else None,
            account_cash_mode_map=account_cash_mode_map,
            db_mtime=db_mtime,
        )

        # 2. Benchmarks mapping
        mapped_benchmarks = []
        ticker_to_name = {}
        if benchmarks and isinstance(benchmarks, (list, set, tuple)):
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    ticker = config.BENCHMARK_MAPPING[b]
                    mapped_benchmarks.append(ticker)
                    ticker_to_name[ticker] = b
                else:
                    mapped_benchmarks.append(b)
                    ticker_to_name[b] = b

        # 3. Fetch Historical Prices for Stock + Benchmarks
        mdp = get_mdp()
        yf_ticker = yf_mapped or sym_clean
        symbols_to_fetch = [yf_ticker] + mapped_benchmarks

        hist_data, _ = await asyncio.to_thread(
            mdp.get_historical_data,
            symbols_to_fetch,
            calc_start_date - timedelta(days=10),
            end_date,
            interval="1d",
        )

        if yf_ticker not in hist_data or hist_data[yf_ticker].empty:
            return []

        stock_df = hist_data[yf_ticker].copy()
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            stock_df.index = pd.to_datetime(stock_df.index, utc=True)
        else:
            stock_df.index = (
                stock_df.index.tz_convert("UTC")
                if stock_df.index.tz is not None
                else stock_df.index.tz_localize("UTC")
            )

        price_col = (
            "price"
            if "price" in stock_df.columns
            else ("Close" if "Close" in stock_df.columns else "Adj Close")
        )
        if price_col not in stock_df.columns:
            return []

        price_adjusted_series = stock_df[price_col].dropna()
        if price_adjusted_series.empty:
            return []

        # Un-adjust historical prices for stock splits so that raw ledger quantities match historical prices
        split_txs = df_sym[
            df_sym["Type"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(["split", "stock split"])
        ]
        factors = pd.Series(1.0, index=price_adjusted_series.index)
        if not split_txs.empty:
            split_txs = split_txs.copy()
            split_txs["__split_priority"] = np.where(
                split_txs["Account"].astype(str).str.lower() == "all accounts", 0, 1
            )
            split_txs["__ym"] = pd.to_datetime(split_txs["Date"]).dt.to_period("M")
            sort_cols = ["Symbol", "__ym", "__split_priority"]
            if "original_index" in split_txs.columns:
                sort_cols.append("original_index")
            split_txs = split_txs.sort_values(by=sort_cols)
            split_txs = split_txs.drop_duplicates(
                subset=["Symbol", "__ym", "Split Ratio"]
            )

            price_dates = pd.Series(
                [ts.date() for ts in price_adjusted_series.index],
                index=price_adjusted_series.index,
            )
            sorted_splits = split_txs.sort_values(by="Date", ascending=False)
            for _, split_row in sorted_splits.iterrows():
                s_date = (
                    split_row["Date"].date()
                    if isinstance(split_row["Date"], pd.Timestamp)
                    else pd.to_datetime(split_row["Date"]).date()
                )
                ratio = pd.to_numeric(split_row.get("Split Ratio"), errors="coerce")
                qty = pd.to_numeric(split_row.get("Quantity"), errors="coerce")
                if (ratio is None or ratio <= 1e-9) and (
                    qty is not None and 0 < qty <= 20.0
                ):
                    ratio = qty
                if ratio and ratio > 1e-9:
                    factors[price_dates < s_date] *= ratio

        raw_price_series = price_adjusted_series * factors

        # 4. FX series for display currency conversion
        target_curr_upper = currency.upper()
        # Find default account currency or from transaction
        local_curr = "USD"
        if (
            "Local Currency" in df_sym.columns
            and not df_sym["Local Currency"].dropna().empty
        ):
            local_curr = str(df_sym["Local Currency"].dropna().iloc[0]).upper().strip()
        elif account_currency_map:
            acc_name = (
                df_sym["Account"].dropna().iloc[0]
                if "Account" in df_sym.columns and not df_sym["Account"].dropna().empty
                else None
            )
            if acc_name and acc_name in account_currency_map:
                local_curr = account_currency_map[acc_name].upper().strip()

        fx_series = pd.Series(1.0, index=raw_price_series.index)
        if local_curr != target_curr_upper and historical_fx_yf:
            fx_pair = f"{target_curr_upper}=X"
            if fx_pair not in historical_fx_yf and target_curr_upper == "THB":
                fx_pair = "USDTHB=X"
            if fx_pair in historical_fx_yf:
                fx_df = historical_fx_yf[fx_pair]
                rate_c = (
                    "price"
                    if "price" in fx_df.columns
                    else ("Close" if "Close" in fx_df.columns else "rate")
                )
                if rate_c in fx_df.columns:
                    s = fx_df[rate_c].copy()
                    s.index = pd.to_datetime(s.index, utc=True)
                    fx_series = (
                        s.reindex(raw_price_series.index).ffill().bfill().fillna(1.0)
                    )

        # 5. Benchmark price series
        bm_series_dict = {}
        for bm_ticker in mapped_benchmarks:
            if bm_ticker in hist_data and not hist_data[bm_ticker].empty:
                bm_df = hist_data[bm_ticker]
                bm_p_col = (
                    "price"
                    if "price" in bm_df.columns
                    else ("Close" if "Close" in bm_df.columns else "Adj Close")
                )
                if bm_p_col in bm_df.columns:
                    bm_s = bm_df[bm_p_col].copy()
                    bm_s.index = pd.to_datetime(bm_s.index, utc=True)
                    bm_series_dict[bm_ticker] = (
                        bm_s.reindex(raw_price_series.index).ffill().bfill()
                    )

        # 6. Sort transactions chronologically
        tx_sorted = df_sym.sort_values(by=["Date", "original_index"]).copy()

        # Build chronological event ledger
        tx_events = []
        for _, row in tx_sorted.iterrows():
            d = (
                row["Date"].date()
                if isinstance(row["Date"], pd.Timestamp)
                else pd.to_datetime(row["Date"]).date()
            )
            t_type = str(row["Type"]).lower().strip()
            qty = float(row.get("Quantity", 0.0) or 0.0)
            price_local = float(row.get("Price/Share", 0.0) or 0.0)
            comm = float(row.get("Commission", 0.0) or 0.0)
            split_ratio = float(row.get("Split Ratio", 1.0) or 1.0)
            tx_events.append(
                {
                    "date": d,
                    "type": t_type,
                    "quantity": qty,
                    "price_local": price_local,
                    "commission": comm,
                    "split_ratio": split_ratio,
                    "account": str(row.get("Account", "")),
                }
            )

        # Step through daily price points and calculate holding state
        daily_records = []
        open_lots_state: List[Dict[str, Any]] = []
        event_idx = 0
        num_events = len(tx_events)

        cum_return_factor = 1.0
        prev_price_adj_display = None
        prev_shares = 0.0

        for dt_ts, p_val in raw_price_series.items():
            cur_date = dt_ts.date()

            # Process all transactions occurring on or before cur_date
            while event_idx < num_events and tx_events[event_idx]["date"] <= cur_date:
                ev = tx_events[event_idx]
                ev_type = ev["type"]
                ev_qty = ev["quantity"]

                if ev_type in ["buy", "transfer in", "receive", "deposit"]:
                    cps = ev["price_local"] + (
                        ev["commission"] / ev_qty if ev_qty > 1e-6 else 0.0
                    )
                    cur_fx = float(fx_series.get(dt_ts, 1.0) or 1.0)
                    open_lots_state.append(
                        {
                            "qty": ev_qty,
                            "cps_local": cps,
                            "fx": cur_fx,
                            "date": ev["date"],
                        }
                    )
                elif ev_type in ["sell", "transfer out", "deliver", "withdrawal"]:
                    rem_to_sell = ev_qty
                    while rem_to_sell > 1e-6 and open_lots_state:
                        first_lot = open_lots_state[0]
                        if first_lot["qty"] <= rem_to_sell + 1e-6:
                            rem_to_sell -= first_lot["qty"]
                            open_lots_state.pop(0)
                        else:
                            first_lot["qty"] -= rem_to_sell
                            rem_to_sell = 0.0
                            break
                elif ev_type in ["split", "stock split"]:
                    ratio = ev["split_ratio"]
                    if ratio > 1e-6:
                        for lot in open_lots_state:
                            lot["qty"] *= ratio
                            lot["cps_local"] /= ratio

                event_idx += 1

            total_shares = sum(lot["qty"] for lot in open_lots_state)
            if total_shares < config.STOCK_QUANTITY_CLOSE_TOLERANCE:
                total_shares = 0.0

            cur_fx = float(fx_series.get(dt_ts, 1.0) or 1.0)
            price_disp = float(p_val) * cur_fx
            mkt_val = total_shares * price_disp

            # Cost basis of open lots
            cost_basis_disp = (
                sum(
                    lot["qty"]
                    * lot["cps_local"]
                    * (cur_fx if local_curr == target_curr_upper else lot["fx"])
                    for lot in open_lots_state
                )
                if total_shares > 0
                else 0.0
            )
            unreal_g = mkt_val - cost_basis_disp if total_shares > 0 else 0.0
            unreal_pct = (
                (unreal_g / cost_basis_disp * 100.0) if cost_basis_disp > 1e-6 else 0.0
            )

            # Daily return tracking for TWR using split-adjusted price
            p_adj_val = float(price_adjusted_series.get(dt_ts, p_val) or p_val)
            price_adj_disp = p_adj_val * cur_fx
            if (
                prev_price_adj_display is not None
                and prev_price_adj_display > 1e-6
                and prev_shares > 1e-6
            ):
                r_daily = (price_adj_disp / prev_price_adj_display) - 1.0
                cum_return_factor *= 1.0 + r_daily

            prev_price_adj_display = price_adj_disp
            prev_shares = total_shares

            rec = {
                "date": cur_date.strftime("%Y-%m-%d"),
                "value": round(mkt_val, 2),
                "cost_basis": round(cost_basis_disp, 2),
                "shares": round(total_shares, 6),
                "unrealized_gain": round(unreal_g, 2),
                "unrealized_gain_pct": round(unreal_pct, 2),
                "_cum_factor": cum_return_factor,
                "_has_shares": total_shares > 0,
            }

            # Add benchmark prices for normalization
            for bm_ticker, bm_s in bm_series_dict.items():
                if dt_ts in bm_s.index:
                    rec[f"_bm_price_{bm_ticker}"] = float(bm_s.get(dt_ts, 0.0) or 0.0)

            daily_records.append(rec)

        if not daily_records:
            return []

        # 7. Slice to display range
        disp_start_str = display_start_date.strftime("%Y-%m-%d")
        disp_records = [r for r in daily_records if r["date"] >= disp_start_str]
        if not disp_records and daily_records:
            disp_records = [daily_records[-1]]

        # 8. Normalize Return % and Benchmarks to start at 0% at the start of the display window
        # Find baseline return factor
        first_holding_rec = next((r for r in disp_records if r["_has_shares"]), None)
        base_factor = (
            first_holding_rec["_cum_factor"]
            if first_holding_rec
            else disp_records[0]["_cum_factor"]
        )

        # Find benchmark base prices
        bm_base_prices = {}
        for bm_ticker in bm_series_dict:
            first_bm_val = next(
                (
                    r.get(f"_bm_price_{bm_ticker}")
                    for r in disp_records
                    if r.get(f"_bm_price_{bm_ticker}")
                    and r.get(f"_bm_price_{bm_ticker}") > 0
                ),
                None,
            )
            if first_bm_val:
                bm_base_prices[bm_ticker] = first_bm_val

        final_result = []
        for r in disp_records:
            ret_pct = 0.0
            if base_factor > 1e-6 and r.get("_cum_factor") is not None:
                ret_pct = (r["_cum_factor"] / base_factor - 1.0) * 100.0

            out_item = {
                "date": r["date"],
                "value": r["value"],
                "cost_basis": r["cost_basis"],
                "shares": r["shares"],
                "unrealized_gain": r["unrealized_gain"],
                "unrealized_gain_pct": r["unrealized_gain_pct"],
                "return_pct": round(ret_pct, 2),
            }

            for bm_ticker, b_base in bm_base_prices.items():
                cur_bm_p = r.get(f"_bm_price_{bm_ticker}")
                display_name = ticker_to_name.get(bm_ticker, bm_ticker)
                if cur_bm_p and b_base > 1e-6:
                    out_item[display_name] = round((cur_bm_p / b_base - 1.0) * 100.0, 2)
                else:
                    out_item[display_name] = 0.0

            final_result.append(out_item)

        return clean_nans(final_result)

    except Exception as e:
        logging.error(
            f"Error getting stock position history for {symbol}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get position history for {symbol}"
        )


@router.get("/history")
async def get_history(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    period: str = "1y",
    benchmarks: Optional[List[str]] = Query(None),
    interval: str = "1d",
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    force: bool = False,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Returns historical portfolio performance (Value and TWR) and benchmarks.
    """
    logging.info(
        f"get_history: period={period}, interval={interval}, from={from_date}, to={to_date}, force={force}"
    )
    try:
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)

        # If every selected account is closed (closure date <= today), cap the
        # graph end_date at the latest closure date so the line doesn't run
        # flat to "today" on accounts that have already been wound down.
        end_date_cap: Optional[date] = None
        try:
            config_manager = get_config_manager(current_user)
            closure_dates_map = (
                config_manager.gui_config.get("account_closure_dates", {}) or {}
            )
        except Exception:
            closure_dates_map = {}
        _, all_selected_closed = compute_account_closure_state(
            accounts, closure_dates_map, date.today()
        )
        if all_selected_closed and accounts:
            parsed_dates: List[date] = []
            for acc in accounts:
                d_str = closure_dates_map.get(acc)
                if not d_str:
                    continue
                try:
                    parsed_dates.append(
                        datetime.strptime(str(d_str), "%Y-%m-%d").date()
                    )
                except (ValueError, TypeError):
                    continue
            if parsed_dates:
                end_date_cap = max(parsed_dates)

        return await _calculate_historical_performance_internal(
            currency=currency,
            period=period,
            accounts=accounts,
            benchmarks=mapped_benchmarks,
            data=data,
            return_df=False,
            interval=interval,
            from_date_str=from_date,
            to_date_str=to_date,
            force=force,
            end_date_cap=end_date_cap,
        )
    except Exception as e:
        logging.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio history")


@router.get("/portfolio_health")
async def get_portfolio_health(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    show_closed: Optional[bool] = Query(None),
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a comprehensive portfolio health score and breakdown.
    """
    try:
        logging.info(f"Health: Fetching summary for accounts: {accounts}")
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            show_closed_positions=show_closed if show_closed is not None else False,
            data=data,
            current_user=current_user,
        )
        summary_df = summary_data.get("summary_df")

        if summary_df is None:
            logging.warning("Health: Summary DF is None")
            summary_df = pd.DataFrame()
        else:
            logging.info(f"Health: Summary DF shape: {summary_df.shape}")

        # 2. Get Risk Metrics (for efficiency/volatility)
        logging.info("Health: Fetching history (1y period)")
        history_df = await _calculate_historical_performance_internal(
            currency=currency,
            period="1y",  # Standard period for health check
            accounts=accounts,
            benchmarks=["S&P 500"],  # Use S&P 500 for Beta/Alpha
            data=data,
            return_df=True,  # Requested DF for calculations
        )

        portfolio_series = pd.Series(dtype=float)
        benchmark_series = pd.Series(dtype=float)
        if (
            history_df is not None
            and not history_df.empty
            and "value" in history_df.columns
        ):
            logging.info(f"Health: History DF shape: {history_df.shape}")
            # Extract portfolio portfolio value series
            history_df_reset = history_df.set_index("date")
            portfolio_series = history_df_reset["value"]

            # Extract benchmark series if available (using ticker ^GSPC which S&P 500 maps to)
            if "^GSPC Price" in history_df_reset.columns:
                benchmark_series = history_df_reset["^GSPC Price"]
        else:
            logging.warning(
                f"Health: History DF is empty or missing 'value'. Columns: {history_df.columns if history_df is not None else 'None'}"
            )

        risk_metrics = calculate_all_risk_metrics(
            portfolio_series,
            benchmark_values=benchmark_series if not benchmark_series.empty else None,
        )

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
            "debug_error": str(e),  # Temporary for debugging
            "components": {
                "diversification": {
                    "score": 0,
                    "metric": 0,
                    "label": f"Err: {str(e)[:20]}",
                },
                "efficiency": {"score": 0, "metric": 0, "label": "Error"},
                "stability": {"score": 0, "metric": "0%", "label": "Error"},
            },
        }
