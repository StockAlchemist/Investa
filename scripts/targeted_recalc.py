
import sys
import os
import logging
import asyncio
from datetime import datetime
import yfinance as yf

# Convert to absolute path to find src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from market_data import get_shared_mdp
from db_utils import get_db_connection, update_intrinsic_value_in_cache
from financial_ratios import get_comprehensive_intrinsic_value
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def targeted_recalc():
    symbols = ["BMY", "EXE", "LUV", "DOC", "EG", "TKO", "LYV"]
    logging.info(f"Targeted refresh for: {symbols}")
    
    db_conn = get_db_connection()
    mdp = get_shared_mdp()
    
    for symbol in symbols:
        try:
            logging.info(f"Processing {symbol}...")
            info = mdp.get_fundamental_data(symbol)
            financials = mdp.get_financials(symbol, "annual")
            balance_sheet = mdp.get_balance_sheet(symbol, "annual")
            cashflow = mdp.get_cashflow(symbol, "annual")
            
            if info:
                valuation = get_comprehensive_intrinsic_value(
                    info, financials, balance_sheet, cashflow
                )
                
                avg_iv = valuation.get('average_intrinsic_value')
                mos = valuation.get('margin_of_safety_pct')
                
                logging.info(f"  {symbol}: New IV=${avg_iv:.22} | New MOS={mos:.2f}%")
                
                update_intrinsic_value_in_cache(
                    db_conn,
                    symbol,
                    intrinsic_value=avg_iv,
                    margin_of_safety=mos,
                    last_fiscal_year_end=info.get("lastFiscalYearEnd"),
                    most_recent_quarter=info.get("mostRecentQuarter"),
                    info=info
                )
        except Exception as e:
            logging.error(f"Failed {symbol}: {e}")
            
    db_conn.close()
    logging.info("Targeted refresh complete.")

if __name__ == "__main__":
    asyncio.run(targeted_recalc())
