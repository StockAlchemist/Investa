
import sys
import os
import logging
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from market_data import MarketDataProvider

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_etf_data(symbol="SCHG"):
    print(f"Testing ETF data extraction for {symbol}...")
    
    # Create temp dir for DB and cache
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir: {temp_dir}")
    
    db_path = os.path.join(temp_dir, "test_market.db")
    fundamentals_cache = os.path.join(temp_dir, "fundamentals_cache")
    
    try:
        # Initialize Provider
        provider = MarketDataProvider(
            db_path=db_path,
            fundamentals_cache_dir=fundamentals_cache
        )
        
        # Fetch data
        print("Fetching fundamental data...")
        data = provider.get_fundamental_data(symbol)
        
        if not data:
            print("No data returned!")
            return

        print("\n--- Extracted Data ---")
        if 'etf_data' in data:
            etf = data['etf_data']
            print("ETF Data Found!")
            
            print(f"Top Holdings ({len(etf.get('top_holdings', []))}):")
            for h in etf.get('top_holdings', [])[:5]:
                print(f"  - {h.get('symbol', 'N/A')}: {h.get('percent', 0)}")
                
            print("\nSector Weightings:")
            for s, w in etf.get('sector_weightings', {}).items():
                print(f"  - {s}: {w}")
                
            print("\nAsset Classes:")
            for a, v in etf.get('asset_classes', {}).items():
                print(f"  - {a}: {v}")
                
            # Verify structure matches API expectations
            if not isinstance(etf.get('top_holdings'), list):
                 print("ERROR: top_holdings is not a list")
            if not isinstance(etf.get('sector_weightings'), dict):
                 print("ERROR: sector_weightings is not a dict")

        else:
            print("No 'etf_data' found in fundamentals.")
            print(f"Exchange: {data.get('exchange')}")
            print(f"QuoteType: {data.get('quoteType')}")
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("\nCleanup done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_etf_data(sys.argv[1])
    else:
        test_etf_data()
