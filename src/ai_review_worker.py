# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          ai_review_worker.py
 Purpose:       Standalone worker to generate AI stock reviews for S&P 500 and S&P 400 companies.
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
from typing import List

# Ensure 'src' is in the python path if running from within src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import necessary modules
try:
    import config
    from market_data import get_shared_mdp
    from server.screener_service import get_sp500_tickers, get_sp400_tickers
    from server.ai_analyzer import generate_stock_review
    from financial_ratios import calculate_key_ratios_timeseries, get_comprehensive_intrinsic_value
    from db_utils import get_db_connection, update_intrinsic_value_in_cache, update_ai_review_in_cache
except ImportError as e:
    print(f"CRITICAL: Failed to import modules. Make sure you are running with 'src' in PYTHONPATH. Error: {e}")
    sys.exit(1)

BATCH_DELAY_SECONDS = 15 # Increased to 15s to respect Gemini 3 Flash 5 RPM limit (60s / 5 = 12s + buffer)
USE_GROUNDING = True # Toggle Google Search (True uses search, False disables it)
MAX_CONSECUTIVE_FAILURES = 5
QUOTA_RESET_HOUR = 15 # 3 PM local time

def sleep_until_next_quota_reset(target_hour=QUOTA_RESET_HOUR):
    """
    Calculates the time until the next target_hour (e.g., 3 PM) and sleeps until then.
    Used when rate limits are hit to wait for the quota reset.
    """
    now = datetime.datetime.now()
    target_time = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    
    # If it's already past target_hour today, target the same hour tomorrow
    if now >= target_time:
        target_time += datetime.timedelta(days=1)
        
    wait_seconds = (target_time - now).total_seconds()
    
    logging.info(f"Quota limit reached or excessive failures. Next reset at: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Sleeping for {wait_seconds/3600:.2f} hours...")
    
    time.sleep(wait_seconds)
    logging.info("Wake up! Resuming work after quota reset.")

def check_if_review_exists(symbol: str, fund_data: dict = None, universe: str = 'sp500') -> bool:
    """
    Checks if a valid AI review already exists for the symbol in the cache for the specified universe.
    Performs smart invalidation based on fiscal year/quarter data.
    """
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        # Direct SQL query to check specific universe cache
        # We need to handle the composite PK (symbol, universe) effectively
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ai_summary, last_fiscal_year_end, most_recent_quarter FROM screener_cache WHERE symbol=? AND universe=?", 
            (symbol, universe)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            summary_text = row[0] if row[0] else ""
            cached_fy_end = row[1]
            cached_mrq = row[2]
            
            has_review = bool(summary_text)
            
            # --- FAILURE DETECTION ---
            if has_review:
                if "error" in summary_text.lower() or len(summary_text) < 20:
                    logging.warning(f"Invalidating cache for {symbol} ({universe}): AI Summary indicates failure or is too short.")
                    return False
            
            if has_review and fund_data:
                # --- SMART INVALIDATION ---
                live_fy_end = fund_data.get("lastFiscalYearEnd")
                live_mrq = fund_data.get("mostRecentQuarter")
                
                # If cached data timestamp identifiers are stale compared to live data,
                # we declare the review as "not existing" to force a refresh.
                if cached_fy_end != live_fy_end or cached_mrq != live_mrq:
                    logging.info(f"Invalidating cache for {symbol} ({universe}): New financial data detected. Cached(FY={cached_fy_end}, Q={cached_mrq}) vs Live(FY={live_fy_end}, Q={live_mrq})")
                    return False
            
            return has_review
        return False
    except Exception as e:
        logging.error(f"Error checking review existence for {symbol}: {e}")
        if conn:
            conn.close()
        return False

def save_review_to_screener(symbol: str, review: dict, fund_data: dict, universe: str = 'sp500'):
    """
    Saves the generated AI review to the screener cache with the specified universe.
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        # update_ai_review_in_cache handles the logic of updating or creating new entry
        update_ai_review_in_cache(
            conn, 
            symbol, 
            review, 
            info=fund_data, 
            universe=universe
        )
        conn.close()
        logging.info(f"Saved review to screener cache for {symbol} (universe='{universe}').")
    except Exception as e:
        logging.error(f"Error saving review to screener for {symbol}: {e}")
        if conn:
            conn.close()

def process_stock(symbol: str, mdp, fund_data: dict, universe: str = 'sp500') -> bool:
    """
    Fetches data and generates review for a single stock.
    Returns True if successful (or skipped), False if failed.
    """
    try:
        logging.info(f"Processing {symbol} ({universe})...")
        
        # 1. Map Symbol
        yf_symbol = symbol 
        
        # 2. Validate Fundamentals
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
        
        # 4. Generate AI review
        try:
            review = generate_stock_review(symbol, fund_data, ratios, use_search=USE_GROUNDING)
        except Exception as e:
            logging.error(f"Failed to generate review for {symbol}: {e}")
            return False # Indicate failure
            
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
                    info=fund_data,
                    universe=universe # Ensure we update the correct universe row
                )
                conn.close()
                logging.info(f"Updated intrinsic value cache for {symbol}.")
        except Exception as iv_e:
            logging.error(f"Failed to calculate/update intrinsic value for {symbol}: {iv_e}")

        # 7. Save Review to Screener Table
        save_review_to_screener(symbol, review, fund_data, universe=universe)
        
        return True

    except Exception as e:
        logging.error(f"Exception processing {symbol}: {e}", exc_info=True)
        return False

def process_ticker_list(mdp, tickers: List[str], universe: str) -> int:
    """
    Processes a list of tickers for a given universe.
    Returns the number of stocks actually processed (generated reviews).
    """
    logging.info(f"Starting processing for universe: {universe} ({len(tickers)} tickers)")
    
    if not tickers:
        logging.warning(f"No tickers provided for {universe}.")
        return 0
        
    processed_count = 0
    consecutive_failures = 0
    
    for symbol in tickers:
        try:
            # 1. Fetch Fundamentals FIRST (Needed for Smart Cache Check)
            yf_symbol = symbol 
            fund_data = mdp.get_fundamental_data(yf_symbol)
            
            # Check cache with smart invalidation
            if check_if_review_exists(symbol, fund_data, universe=universe):
                # logging.info(f"Skipping {symbol} - Valid & Fresh review exists.")
                continue
            
            logging.info(f"Review needed for {symbol} in {universe}. Starting generation.")
            success = process_stock(symbol, mdp, fund_data, universe=universe)
            
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
                    # Wait until 3 PM local time for the next quota reset
                    sleep_until_next_quota_reset()
                    consecutive_failures = 0 # Reset after long sleep
                    # We break to refresh the list and start fresh from the beginning
                    return processed_count
                    
        except KeyboardInterrupt:
            logging.info("Worker stopped by user.")
            sys.exit(0)
        except Exception as e:
             logging.error(f"Error iterating {symbol}: {e}")
             
    logging.info(f"Finished pass for {universe}. Processed {processed_count} stocks.")
    return processed_count

def main():
    # Configure logging to ensure output is visible
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting AI Review Worker (S&P 500 & S&P 400)...")
    
    # Get Market Data Provider
    mdp = get_shared_mdp()
    
    while True:
        try:
            total_processed = 0
            
            # 1. Process S&P 500
            logging.info("Fetching S&P 500 list...")
            sp500_tickers = get_sp500_tickers()
            total_processed += process_ticker_list(mdp, sp500_tickers, "sp500")
            
            # 2. Process S&P 400
            logging.info("Fetching S&P 400 list...")
            sp400_tickers = get_sp400_tickers()
            total_processed += process_ticker_list(mdp, sp400_tickers, "sp400")

            if total_processed == 0:
                logging.info("No stocks needed processing (all cached). Sleeping for 1 hour before re-checking...")
                time.sleep(3600)
            else:
                logging.info(f"Cycle complete. Processed {total_processed} stocks. Sleeping for 1 hour...")
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
