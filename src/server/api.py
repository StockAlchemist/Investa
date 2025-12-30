from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd
import logging
import os
from datetime import datetime, date # Added this import



from server.dependencies import get_transaction_data, get_config_manager, reload_data
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
from portfolio_analyzer import calculate_periodic_returns, extract_realized_capital_gains_history, extract_dividend_history
from market_data import MarketDataProvider, map_to_yf_symbol
from db_utils import add_transaction_to_db, update_transaction_in_db, delete_transaction_from_db, get_database_path, get_db_connection


from risk_metrics import calculate_all_risk_metrics, calculate_drawdown_series
import config
from pydantic import BaseModel, Field

router = APIRouter()

current_file_path = os.path.abspath(__file__)
src_server_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(src_server_dir)
project_root = os.path.dirname(src_dir)

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

async def _calculate_portfolio_summary_internal(
    currency: str = "USD",
    include_accounts: Optional[List[str]] = None,
    show_closed_positions: bool = True,
    data: tuple = None
) -> Dict[str, Any]:
    """Internal helper to calculate portfolio summary data."""
    (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        _
    ) = data

    if df.empty:
        return {"metrics": {}, "rows": []}

    mdp = MarketDataProvider()
    (
        overall_summary_metrics,
        summary_df,
        holdings_dict,  # Unpack holdings dict
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
        market_provider=mdp
    )

    return {
        "metrics": overall_summary_metrics,
        "summary_df": summary_df,
        "holdings_dict": holdings_dict,
        "account_metrics": account_level_metrics
    }

@router.get("/summary")
async def get_portfolio_summary(
    currency: str = "USD",
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
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
        original_csv_path
    ) = data
    
    if df.empty:
        return {"error": "No transaction data available"}

    try:
        # Calculate portfolio summary using helper
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            data=data
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
        original_csv_path
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
            show_closed_positions=False, # Don't show closed in holdings
            data=data
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
    
    class Config:
        populate_by_name = True

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
        _, _, _, _, _, db_path = data
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

        success, new_id = add_transaction_to_db(conn, tx_data)
        conn.close()
        
        if success:
            reload_data() # Refresh cache
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
        _, _, _, _, _, db_path = data
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

        success = update_transaction_in_db(conn, transaction_id, tx_data)
        conn.close()
        
        if success:
            reload_data()
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
        transaction_id (int): The ID of the transaction to delete.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message.
    """
    try:
        _, _, _, _, _, db_path = data
        conn = get_db_connection(db_path)
        if not conn:
             raise HTTPException(status_code=500, detail="Database connection failed")
             
        success = delete_transaction_from_db(conn, transaction_id)
        conn.close()
        
        if success:
            reload_data()
            return {"status": "success", "message": "Transaction deleted"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found or delete failed")

    except Exception as e:
        logging.error(f"Error deleting transaction: {e}", exc_info=True)
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

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names.
        period (str): Time period (e.g., '1y', 'ytd', 'all').
        benchmarks (List[str], optional): List of benchmark symbols.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of daily records containing date, portfolio value, TWR, and benchmark performance.
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

        # Prepare reverse mapping for display names
        ticker_to_name = {v: k for k, v in config.BENCHMARK_MAPPING.items()}

        # Calculate Drawdown Series
        drawdown_series = calculate_drawdown_series(daily_df["Portfolio Value"])

        # Format for frontend
        result = []
        # daily_df index is Date
        for dt, row in daily_df.iterrows():
            # Handle NaN values
            val = row.get("Portfolio Value", 0.0)
            twr = row.get("Portfolio Accumulated Gain", 1.0)
            dd = drawdown_series.get(dt, 0.0)
            
            item = {
                "date": dt.strftime("%Y-%m-%d"),
                "value": val if pd.notnull(val) else 0.0,
                "twr": (twr - 1) * 100 if pd.notnull(twr) else 0.0, # Convert to percentage change
                "drawdown": dd * 100 if pd.notnull(dd) else 0.0 # Convert to percentage
            }

            # Add benchmark data
            for b_ticker in mapped_benchmarks:
                if b_ticker in daily_df.columns:
                     b_val = row.get(b_ticker)
                     display_name = ticker_to_name.get(b_ticker, b_ticker)
                     item[display_name] = (b_val - 1) * 100 if pd.notnull(b_val) else 0.0
                
            result.append(item)
            
        return clean_nans(result)

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

    Args:
        currency (str): The display currency.
        accounts (List[str], optional): List of account names.
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
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
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

        return {
            "manual_overrides": enriched_overrides,
            "user_symbol_map": user_symbol_map,
            "user_excluded_symbols": list(user_excluded_symbols),
            "account_currency_map": account_currency_map
        }
    except Exception as e:
        logging.error(f"Error getting settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class SettingsUpdate(BaseModel):
    manual_price_overrides: Optional[Dict[str, Any]] = None
    user_symbol_map: Optional[Dict[str, str]] = None
    user_excluded_symbols: Optional[List[str]] = None

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
             # If sending a full dict, replace. If we want partial update, we need logic.
             # Let's assume the UI sends the FULL set of overrides for that category usually,
             # OR we implement partial update logic.
             # Given "add option... consistency", usually these UIs show a list and you save.
             # For safety with concurrent edits (unlikely here), let's just merge specific keys if provided,
             # BUT simpler is to let frontend manage the state and send the full dict for that section.
             # However, ConfigManager.save_manual_overrides expects the FULL structure of all 3 categories
             # passed as one dict, OR it updates `self.manual_overrides` with the dict passed.
             
             # Let's see ConfigManager.save_manual_overrides:
             # if overrides_data is not None: self.manual_overrides = overrides_data
             
             # So we need to construct the full new state.
             
             new_price_overrides = current_overrides.get("manual_price_overrides", {}).copy()
             # We can't just merge if the user DELETED something.
             # So if the user sends `manual_price_overrides` we should probably treat it as the "new state" for that key,
             # i.e. REPLACE the `manual_price_overrides` section, but keep others.
             new_price_overrides = settings.manual_price_overrides
             
             current_overrides["manual_price_overrides"] = new_price_overrides

        if settings.user_symbol_map is not None:
            current_overrides["user_symbol_map"] = settings.user_symbol_map
            
        if settings.user_excluded_symbols is not None:
            current_overrides["user_excluded_symbols"] = sorted(list(set(settings.user_excluded_symbols)))
            
        # Save to AppData
        if config_manager.save_manual_overrides(current_overrides):
            
            # --- Added: Mirror save to Project Root if file exists ---
            # This ensures the user's "source of truth" in their workspace stays in sync
            project_overrides_file = os.path.join(project_root, "manual_overrides.json")
            if os.path.exists(project_overrides_file):
                try:
                    import json
                    with open(project_overrides_file, "w", encoding="utf-8") as f:
                        json.dump(current_overrides, f, indent=4, ensure_ascii=False)
                    logging.info(f"Mirrored settings update to {project_overrides_file}")
                except Exception as e:
                    logging.warning(f"Failed to mirror settings to project file: {e}")
            # ---------------------------------------------------------

            # Reload data to apply changes (clear cache)
            reload_data()
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
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, _ = data
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
            interval="D"
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
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns performance attribution by sector and stock.
    """
    df, manual_overrides, user_symbol_map, user_excluded_symbols, account_currency_map, _ = data
    if df.empty:
        return {}

    try:
        # Get current summary rows which contain gains and sector info
        summary_data = await _calculate_portfolio_summary_internal(
            currency=currency,
            include_accounts=accounts,
            data=data
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

        return clean_nans({
            "sectors": sector_attribution,
            "stocks": stock_data[:10], # Top 10 contributors/detractors
            "total_gain": total_gain
        })
    except Exception as e:
        logging.error(f"Error calculating attribution: {e}")
        return {"error": str(e)}

@router.get("/dividend_calendar")
async def get_dividend_calendar(
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns upcoming dividend events for the portfolio.
    """
    df, _, user_symbol_map, user_excluded_symbols, _, _ = data
    if df.empty:
        return []

    try:
        # Get current holdings symbols
        summary_data = await _calculate_portfolio_summary_internal(include_accounts=accounts, show_closed_positions=False, data=data)
        summary_df = summary_data.get("summary_df")
        if summary_df is None or summary_df.empty:
             return []
        rows = summary_df.to_dict(orient="records")
        symbols = [r["Symbol"] for r in rows if r["Symbol"] != "Total" and not r.get("is_total")]
        
        if not symbols:
            return []

        # Use MarketDataProvider to get dividend info
        provider = MarketDataProvider()
        yf_symbols = set()
        from finutils import is_cash_symbol # Import here to avoid circular if top-level issues, though api imports finutils at top
        
        for s in symbols:
            if is_cash_symbol(s):
                continue
            yf_sym = map_to_yf_symbol(s, user_symbol_map, user_excluded_symbols)
            if yf_sym:
                yf_symbols.add(yf_sym)
        
        # ------------------------------------------------------------------
        # We need to fetch Ticker.calendar for each symbol
        import yfinance as yf
        calendar_events = []
        
        stock_data_map = {} # To cache correct symbol casing if needed
        # Create map of symbol -> quantity
        symbol_quantity_map = {}
        for r in rows:
            if r["Symbol"] != "Total" and not r.get("is_total"):
                 symbol_quantity_map[r["Symbol"]] = r["Quantity"]

        # ...
        
        for sym in yf_symbols:
            try:
                t = yf.Ticker(sym)
                cal = t.calendar
                if cal and 'Dividend Date' in cal:
                    # Calculate estimated total payment
                    # dividendRate is Annual. lastDividendValue is usually the single payment amount.
                    per_share_amt = t.info.get('lastDividendValue', 0)
                    if per_share_amt is None or per_share_amt == 0:
                        # Fallback: estimate from annual rate (assuming quarterly)
                        # This is a rough fallback
                        # print(f"DEBUG: {sym} lastDividendValue missing, using dividendRate/4")
                        per_share_amt = t.info.get('dividendRate', 0) / 4.0
                    
                    # Find quantity
                    qty = 0
                    # Reconstruct map
                    yf_to_orig = {}
                    for s in symbols:
                        mapped = map_to_yf_symbol(s, user_symbol_map, user_excluded_symbols)
                        if mapped:
                            yf_to_orig[mapped] = s
                    
                    orig_sym = yf_to_orig.get(sym)
                    if orig_sym:
                        qty = symbol_quantity_map.get(orig_sym, 0)

                    calendar_events.append({
                        "symbol": orig_sym if orig_sym else sym,
                        "dividend_date": str(cal['Dividend Date']),
                        "ex_dividend_date": str(cal.get('Ex-Dividend Date', '')),
                        "amount": (per_share_amt if per_share_amt else 0) * qty
                    })
            except Exception as e_cal:
                logging.warning(f"Failed to fetch calendar for {sym}: {e_cal}")

        return clean_nans(calendar_events)
    except Exception as e:
        logging.error(f"Error fetching dividend calendar: {e}")
        return {"error": str(e)}

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
        reload_data()
        
        return {"status": "success", "message": f"Override for {symbol_upper} updated."}

    except Exception as e:
        logging.error(f"Error updating manual override: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class WebhookRefreshRequest(BaseModel):
    secret: str

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
        reload_data()
        
        return {"status": "success", "message": "Market data cache invalidated and data reloaded."}
        
    except Exception as e:
        logging.error(f"Error in webhook refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
