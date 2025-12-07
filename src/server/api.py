from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime, date # Added this import

print("DEBUG: LOADING API MODULE", flush=True) # Added this line

from server.dependencies import get_transaction_data
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
from portfolio_analyzer import calculate_periodic_returns, extract_realized_capital_gains_history, extract_dividend_history
from market_data import MarketDataProvider

router = APIRouter()

# ... (existing code)

@router.get("/asset_change")
async def get_asset_change(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns periodic asset change data (Annual, Monthly, Weekly, Daily).
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
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
            interval="D"
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
                p_df_reset = p_df.reset_index()
                # Convert dates to strings
                if 'Date' in p_df_reset.columns:
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
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj

@router.get("/summary")
async def get_portfolio_summary(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the high-level portfolio summary (Total Value, G/L, etc.).
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
    ) = data
    
    if df.empty:
        return {"error": "No transaction data available"}

    try:
        # Calculate portfolio summary
        import sys

        # Instantiate MDP to ensure consistent data access
        mdp = MarketDataProvider()
        
        (
            overall_summary_metrics,
            summary_df,
            account_level_metrics,
            combined_ignored_indices,
            combined_ignored_reasons,
            status_msg
        ) = calculate_portfolio_summary(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            fmp_api_key=getattr(config, "FMP_API_KEY", None),
            display_currency=currency,
            show_closed_positions=True,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            include_accounts=accounts,
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            market_provider=mdp
        )

        # --- Calculate Annualized TWR ---
        annualized_twr = None
        try:
            # Determine date range for "all" time
            if not df.empty:
                min_date = df["Date"].min().date()
                max_date = date.today()
                
                # Fetch full history to calculate TWR over the entire period
                # Note: calculate_historical_performance requires start_date, end_date, etc.
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
                    user_excluded_symbols=user_excluded_symbols
                )
                
                if daily_df is not None and not daily_df.empty:
                    twr_col = "Portfolio Accumulated Gain"
                    if twr_col in daily_df.columns:
                        # Get the final TWR factor (cumulative)
                        # daily_df[twr_col] contains the cumulative TWR factor (e.g., 1.5 for 50% gain)
                        final_twr_factor = daily_df[twr_col].iloc[-1]
                        
                        # Use the actual data range from the result for accuracy
                        res_start_date = daily_df.index[0].date()
                        res_end_date = daily_df.index[-1].date()
                        days = (res_end_date - res_start_date).days
                        
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
                mdp = MarketDataProvider()
                indices_data = mdp.get_index_quotes(config.INDICES_FOR_HEADER)
                overall_summary_metrics["indices"] = indices_data
            except Exception as e_indices:
                logging.warning(f"Failed to fetch market indices: {e_indices}")

        
        response_data = {
            "metrics": overall_summary_metrics,
            "account_metrics": account_level_metrics
        }
        return clean_nans(response_data)
    except Exception as e:
        logging.error(f"Error calculating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/holdings")
async def get_holdings(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the list of current holdings.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
    ) = data
    
    if df.empty:
        return []

    try:
        (
            _,
            summary_df,
            _,
            _, _, _
        ) = calculate_portfolio_summary(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            display_currency=currency,
            show_closed_positions=False,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
            include_accounts=accounts
        )
        
        if summary_df is None or summary_df.empty:
            return []

        # Convert DataFrame to list of dicts
        # Handle NaNs
        summary_df = summary_df.where(pd.notnull(summary_df), None)
        
        # We need to make sure we return a clean list of dicts
        records = summary_df.to_dict(orient="records")
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
    """
    df, _, _, _, _, _ = data
    
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
        records = df.to_dict(orient="records")
        return clean_nans(records)
        
    except Exception as e:
        logging.error(f"Error getting transactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    period: str = "1y",
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns historical portfolio performance (Value and TWR) and benchmarks.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
    ) = data

    if df.empty:
        return []

    try:
        # Determine date range
        end_date = date.today()
        if period == "1m":
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
        elif period == "ytd":
            start_date = date(end_date.year, 1, 1)
        elif period == "all":
            start_date = df["Date"].min().date()
        else:
            start_date = end_date - timedelta(days=365)

        # Map display names to tickers if needed, or assume tickers are passed.
        # The frontend should pass tickers or we map them here.
        # Let's assume frontend passes tickers for now, or use config mapping if they pass names.
        # For robustness, let's try to map names to tickers if they match keys in config.
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)

        print(f"DEBUG API: History Request: period={period}, benchmarks={benchmarks}")
        print(f"DEBUG API: History Date Range: start={start_date}, end={end_date}")

        # Call calculation logic
        daily_df, _, _, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=None, # Not needed for web view
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=start_date,
            end_date=end_date,
            interval="D",
            benchmark_symbols_yf=mapped_benchmarks, 
            display_currency=currency,
            account_currency_map=account_currency_map,
            default_currency="USD", # Should come from config but hardcoded for now
            include_accounts=accounts,
            user_symbol_map=user_symbol_map,
            manual_overrides_dict=manual_overrides,
            user_excluded_symbols=user_excluded_symbols,
            original_csv_file_path=original_csv_path
        )

        logging.info(f"API History Result: {len(daily_df)} rows returned from calc.")

        if daily_df.empty:
            return []

        # --- Calculate Benchmark TWR ---
        # daily_df contains "{ticker} Price" columns. We need to calculate TWR for them.
        for b_ticker in mapped_benchmarks:
            price_col = f"{b_ticker} Price"
            if price_col in daily_df.columns:
                # Calculate daily returns
                # pct_change() gives return from prev day. First day is NaN.
                daily_rets = daily_df[price_col].pct_change().fillna(0.0)
                
                # Calculate cumulative return (TWR), starting at 1.0
                # cumprod() propagates NaNs. If we have NaNs in price, we might have NaNs here.
                # We fill initial NaN from pct_change with 0.0 so first day is 1.0.
                daily_df[b_ticker] = (1 + daily_rets).cumprod()

        # Format for frontend
        result = []
        # daily_df index is Date
        for dt, row in daily_df.iterrows():
            # Handle NaN values
            val = row.get("Portfolio Value", 0.0)
            twr = row.get("Portfolio Accumulated Gain", 1.0)
            
            item = {
                "date": dt.strftime("%Y-%m-%d"),
                "value": val if pd.notnull(val) else 0.0,
                "twr": (twr - 1) * 100 if pd.notnull(twr) else 0.0 # Convert to percentage change
            }

            # Add benchmark data
            for b_ticker in mapped_benchmarks:
                # We now expect b_ticker column to exist with TWR values
                if b_ticker in daily_df.columns:
                     b_val = row.get(b_ticker)
                     item[b_ticker] = (b_val - 1) * 100 if pd.notnull(b_val) else 0.0
                
            result.append(item)
            
        return result

    except Exception as e:
        logging.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capital_gains")
async def get_capital_gains(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the realized capital gains history.
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
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
        
        capital_gains_df = extract_realized_capital_gains_history(
            all_transactions_df=df,
            display_currency=currency,
            historical_fx_yf=historical_fx_yf,
            default_currency=config.DEFAULT_CURRENCY,
            shortable_symbols=config.SHORTABLE_SYMBOLS,
            include_accounts=accounts
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
    """
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        original_csv_path
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
@router.get("/settings")
async def get_settings(
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the current application configuration settings.
    """
    (
        _,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        _
    ) = data

    try:
        return {
            "manual_overrides": manual_overrides,
            "user_symbol_map": user_symbol_map,
            "user_excluded_symbols": list(user_excluded_symbols),
            "account_currency_map": account_currency_map
        }
    except Exception as e:
        logging.error(f"Error getting settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
