
import sys
import os
import logging
import asyncio
from datetime import datetime
import pandas as pd
import yfinance as yf

# Convert to absolute path to find src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from server.screener_service import get_sp500_tickers
from db_utils import get_db_connection, update_intrinsic_value_in_cache
from financial_ratios import get_comprehensive_intrinsic_value
from market_data import MarketDataProvider # Added MDP
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sp500_recalc.log"),
        logging.StreamHandler()
    ]
)

async def recalculate_sp500_valuations():
    """
    Fetches S&P 500 tickers and recalculates intrinsic value for each.
    """
    logging.info("Starting S&P 500 Intrinsic Value Recalculation...")
    
    try:
        # Get S&P 500 list
        tickers = get_sp500_tickers()
        logging.info(f"Found {len(tickers)} S&P 500 tickers.")
        
        # Initialize MDP to use centralized caching
        mdp = MarketDataProvider()
        
        db_conn = get_db_connection()
        if not db_conn:
            logging.critical("Could not connect to database.")
            return

        for i, symbol in enumerate(tickers):
            try:
                logging.info(f"[{i+1}/{len(tickers)}] Processing {symbol}...")
                
                # Fetch data via MDP to update the shared JSON fundamentals cache
                # We use force_refresh=True for fundamentals to ensure the JSON cache 
                # matches what we use for calculation.
                info = mdp.get_fundamental_data(symbol, force_refresh=True)
                financials_df = mdp.get_financials(symbol, period="annual", force_refresh=True)
                balance_sheet_df = mdp.get_balance_sheet(symbol, period="annual", force_refresh=True)
                cashflow_df = mdp.get_cashflow(symbol, period="annual", force_refresh=True)
                
                if info and financials_df is not None and not financials_df.empty:
                    # Calculate Intrinsic Value
                    valuation = get_comprehensive_intrinsic_value(
                        info, 
                        financials_df, 
                        balance_sheet_df, 
                        cashflow_df
                    )
                    
                    avg_iv = valuation.get('average_intrinsic_value')
                    mos = valuation.get('margin_of_safety_pct')
                    models = valuation.get('models', {})
                    
                    # Log result
                    model_dcf = models.get('dcf', {}).get('model')
                    model_graham = models.get('graham', {}).get('model')
                    logging.info(f"  > {symbol}: Val=${avg_iv:.2f} | DCF: {model_dcf} | Graham: {model_graham}")
                    
                    # Update DB including full valuation details and metadata
                    info["valuation_details"] = valuation # Inject results for db_utils to pick up
                    update_intrinsic_value_in_cache(
                        db_conn,
                        symbol,
                        intrinsic_value=avg_iv,
                        margin_of_safety=mos,
                        last_fiscal_year_end=info.get("lastFiscalYearEnd"),
                        most_recent_quarter=info.get("mostRecentQuarter"),
                        info=info 
                    )
                    
                else:
                    logging.warning(f"  > {symbol}: Missing data (Info or Financials).")
                    
            except Exception as e:
                logging.error(f"  > {symbol}: Failed - {e}")
                
            # Rate limiting / Sleep to yield
            await asyncio.sleep(0.5)
            
    except Exception as e:
        logging.critical(f"Script failed: {e}")
    finally:
        if db_conn:
            db_conn.close()
        logging.info("Recalculation complete.")

if __name__ == "__main__":
    asyncio.run(recalculate_sp500_valuations())
