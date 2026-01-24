import sys
import os
import logging
import time
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import config
from config_manager import ConfigManager
from market_data import get_shared_mdp
from financial_ratios import get_intrinsic_value_for_symbol
from db_utils import get_db_connection, get_all_watchlists, get_watchlist, update_intrinsic_value_in_cache
from server.screener_service import get_sp500_tickers

def recalculate_all():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting Global Intrinsic Value Recalculation...")
    
    mdp = get_shared_mdp()
    cm = ConfigManager(config.get_app_data_dir())
    db_conn = get_db_connection()
    
    if not db_conn:
        logging.error("Failed to connect to database.")
        return

    all_symbols = set()
    
    # 1. Get S&P 500 tickers
    try:
        sp500 = get_sp500_tickers()
        logging.info(f"Found {len(sp500)} S&P 500 tickers.")
        all_symbols.update(sp500)
    except Exception as e:
        logging.error(f"Error fetching S&P 500: {e}")

    # 2. Get all watchlist tickers
    try:
        watchlists = get_all_watchlists(db_conn)
        logging.info(f"Found {len(watchlists)} watchlists.")
        for wl in watchlists:
            wl_id = wl.get('id')
            if wl_id:
                items = get_watchlist(db_conn, wl_id)
                wl_symbols = [item['Symbol'] for item in items if 'Symbol' in item]
                logging.info(f"Watchlist '{wl.get('name')}': {len(wl_symbols)} symbols.")
                all_symbols.update(wl_symbols)
    except Exception as e:
        logging.error(f"Error fetching watchlists: {e}")

    logging.info(f"Total unique symbols to process: {len(all_symbols)}")
    
    success_count = 0
    fail_count = 0
    
    # 3. Process each symbol
    symbols_list = sorted(list(all_symbols))
    for i, symbol in enumerate(symbols_list):
        try:
            logging.info(f"[{i+1}/{len(symbols_list)}] Recalculating {symbol}...")
            
            # Use centralized high-quality logic
            results = get_intrinsic_value_for_symbol(symbol, mdp, cm)
            
            if "error" in results:
                logging.warning(f"  > Failed {symbol}: {results['error']}")
                fail_count += 1
                continue
            
            # We need info for cache sync timestamps
            info = mdp.get_fundamental_data(symbol)
            
            # Update cache
            if info:
                info["valuation_details"] = results
                
            update_intrinsic_value_in_cache(
                db_conn,
                symbol,
                results.get("average_intrinsic_value"),
                results.get("margin_of_safety_pct"),
                info.get("lastFiscalYearEnd") if info else None,
                info.get("most_recent_quarter") if info else None,
                info=info
            )
            
            iv = results.get('average_intrinsic_value')
            mos = results.get('margin_of_safety_pct')
            
            # Use safe formatting to avoid NoneType errors
            iv_str = f"${iv:.2f}" if iv is not None else "N/A"
            mos_str = f"{mos:.2f}%" if mos is not None else "N/A"
            
            success_count += 1
            logging.info(f"  > Success {symbol}: IV={iv_str} | MOS={mos_str}")
            if results.get("valuation_note"):
                logging.info(f"    [NOTE]: {results['valuation_note']}")
            
        except Exception as e:
            logging.error(f"  > Unexpected error for {symbol}: {e}")
            fail_count += 1
            
    db_conn.close()
    logging.info("\n--- Global Recalculation Summary ---")
    logging.info(f"Successfully processed (Success): {success_count}")
    logging.info(f"Failed to process (Fail): {fail_count}")
    logging.info(f"Total processed in loop: {success_count + fail_count}")
    logging.info(f"Original unique list size: {len(all_symbols)}")

if __name__ == "__main__":
    recalculate_all()
