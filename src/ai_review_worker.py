# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          ai_review_worker.py
 Purpose:       Standalone worker to generate AI stock reviews for S&P 500 companies.
                Runs independently from the main app, checks cache, and handles rate limits.

 Usage:         Run from the 'src' directory or with 'src' in PYTHONPATH.
                python ai_review_worker.py

 Author:        Google Gemini

 Copyright:     (c) Investa Contributors 2026
 Licence:       MIT
-------------------------------------------------------------------------------
"""
import time
import logging
import os
import sys
import datetime
import random

# Ensure 'src' is in the python path if running from within src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import necessary modules
try:
    import config
    from market_data import get_shared_mdp, map_to_yf_symbol
    from server.screener_service import get_sp500_tickers
    from server.ai_analyzer import generate_stock_review
    from financial_ratios import calculate_key_ratios_timeseries, get_comprehensive_intrinsic_value
    from db_utils import get_db_connection, upsert_screener_results, update_intrinsic_value_in_cache
except ImportError as e:
    print(f"CRITICAL: Failed to import modules. Make sure you are running with 'src' in PYTHONPATH. Error: {e}")
    sys.exit(1)

# ... (Previous code)

def process_stock(symbol: str, mdp) -> bool:
    """
    Fetches data and generates review for a single stock.
    Returns True if successful (or skipped), False if failed.
    """
    try:
        logging.info(f"Processing {symbol}...")
        
        # 1. Map Symbol (S&P 500 uses simple tickers, but good to be safe)
        yf_symbol = symbol 
        # Note: get_sp500_tickers returns sanitized separate tickers usually
        
        # 2. Fetch Fundamentals
        fund_data = mdp.get_fundamental_data(yf_symbol)
        if not fund_data:
            logging.warning(f"Skipping {symbol}: No fundamental data found.")
            return True # Treat as "done" so we don't retry endlessly
            
        # 3. Fetch Financials for Ratios AND Intrinsic Value
        financials_df = mdp.get_financials(yf_symbol, "annual")
        balance_sheet_df = mdp.get_balance_sheet(yf_symbol, "annual")
        cashflow_df = mdp.get_cashflow(yf_symbol, "annual")
        
        ratios = {}
        if financials_df is not None and not financials_df.empty and balance_sheet_df is not None and not balance_sheet_df.empty:
            try:
                ratios_df = calculate_key_ratios_timeseries(financials_df, balance_sheet_df)
                if not ratios_df.empty:
                    ratios = ratios_df.iloc[0].to_dict()
            except Exception as e:
                logging.warning(f"Ratio calculation failed for {symbol}: {e}")
        
        # 4. Generate Review
        # force_refresh=False will use cache if it exists, but we did a pre-check.
        # However, passing False is safer.
        review = generate_stock_review(symbol, fund_data, ratios, force_refresh=False)
        
        if "error" in review:
            logging.error(f"Failed to generate review for {symbol}: {review['error']}")
            return False
            
        logging.info(f"Successfully generated/retrieved review for {symbol}.")
        
        # 5. Calculate Intrinsic Value (Detailed Mode)
        try:
            logging.info(f"Calculating intrinsic value for {symbol}...")
            iv_results = get_comprehensive_intrinsic_value(
                fund_data, financials_df, balance_sheet_df, cashflow_df
            )
            
            # 6. Update Cache with Intrinsic Value
            conn = get_db_connection()
            if conn:
                update_intrinsic_value_in_cache(
                    conn,
                    symbol,
                    iv_results.get("average_intrinsic_value"),
                    iv_results.get("margin_of_safety_pct"),
                    fund_data.get("lastFiscalYearEnd"),
                    fund_data.get("mostRecentQuarter"),
                    info=fund_data
                )
                conn.close()
                logging.info(f"Updated intrinsic value cache for {symbol}.")
        except Exception as iv_e:
            logging.error(f"Failed to calculate/update intrinsic value for {symbol}: {iv_e}")

        # 7. Save Review to Screener Table
        save_review_to_screener(symbol, review, fund_data)
        
        return True

    except Exception as e:
        logging.error(f"Exception processing {symbol}: {e}", exc_info=True)
        return False

def main():
    logging.info("Starting AI Review Worker...")
    
    # Get Market Data Provider
    mdp = get_shared_mdp()
    
    consecutive_failures = 0
    
    while True:
        try:
            # 1. Get S&P 500 List
            logging.info("Fetching S&P 500 list...")
            tickers = get_sp500_tickers()
            logging.info(f"Found {len(tickers)} tickers.")
            
            if not tickers:
                logging.error("No tickers found. Retrying in 1 hour.")
                time.sleep(3600)
                continue
                
            # Randomize order to avoid getting stuck on same stocks or alphabetical bias
            # causing issues if we restart
            # random.shuffle(tickers) 
            # Actually, maybe sequential is better to track progress? 
            # But the user asked to "skip those that already have reviews", so order doesn't matter much.
            # Let's keep original order but maybe shuffle slightly or just iterate.
            
            processed_count = 0
            
            for symbol in tickers:
                # check cache first to avoid API calls
                if check_if_review_exists(symbol):
                    # logging.info(f"Skipping {symbol} - Valid review exists.")
                    continue
                
                logging.info(f"Review needed for {symbol}. Starting generation.")
                success = process_stock(symbol, mdp)
                
                if success:
                    consecutive_failures = 0
                    processed_count += 1
                    # Sleep to be polite
                    time.sleep(BATCH_DELAY_SECONDS)
                else:
                    consecutive_failures += 1
                    logging.warning(f"Failure count: {consecutive_failures}")
                    
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logging.error(f"Hit max consecutive failures ({MAX_CONSECUTIVE_FAILURES}). Likely Rate Limit or Network Issue.")
                        logging.info(f"Sleeping for 24 hours (started at {datetime.datetime.now()})...")
                        time.sleep(RATE_LIMIT_WAIT_SECONDS)
                        consecutive_failures = 0 # Reset after long sleep
                        break # Break inner loop to refresh ticker list after sleep
            
            if processed_count == 0:
                logging.info("No stocks needed processing (all cached). Sleeping for 1 hour before re-checking...")
                time.sleep(3600)
            else:
                logging.info("Finished pass through S&P 500. Sleeping for 1 hour...")
                time.sleep(3600)

        except KeyboardInterrupt:
            logging.info("Worker stopped by user.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Global worker exception: {e}", exc_info=True)
            logging.info("Sleeping 5 minutes before restart...")
            time.sleep(300)

if __name__ == "__main__":
    main()
