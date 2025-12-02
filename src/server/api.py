from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

from server.dependencies import get_transaction_data
from portfolio_logic import calculate_portfolio_summary
import config

router = APIRouter()

import math

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
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the high-level portfolio summary (Total Value, G/L, etc.).
    """
    df, ignored_indices, ignored_reasons = data
    
    if df.empty:
        return {"error": "No transaction data available"}

    try:
        (
            overall_metrics,
            summary_df,
            account_metrics,
            _, _, _
        ) = calculate_portfolio_summary(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=ignored_indices,
            ignored_reasons_from_load=ignored_reasons,
            display_currency=currency,
            show_closed_positions=False
        )
        
        response_data = {
            "metrics": overall_metrics,
            "account_metrics": account_metrics
        }
        return clean_nans(response_data)
    except Exception as e:
        logging.error(f"Error calculating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/holdings")
async def get_holdings(
    currency: str = "USD",
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns the list of current holdings.
    """
    df, ignored_indices, ignored_reasons = data
    
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
            ignored_indices_from_load=ignored_indices,
            ignored_reasons_from_load=ignored_reasons,
            display_currency=currency,
            show_closed_positions=False
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
