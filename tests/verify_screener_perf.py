import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from server.screener_service import screen_stocks

def verify_performance():
    logging.basicConfig(level=logging.INFO)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"]
    print(f"Testing screener with {len(symbols)} symbols...")
    
    start_time = time.time()
    results = screen_stocks("manual", manual_symbols=symbols)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nDuration: {duration:.2f} seconds")
    print(f"Items processed: {len(results)}")
    
    if results:
        print("\nSample result (AAPL):")
        aapl = next((r for r in results if r['symbol'] == "AAPL"), None)
        if aapl:
            print(f"  Price: {aapl.get('price')}")
            print(f"  Intrinsic Value: {aapl.get('intrinsic_value')}")
            print(f"  MOS: {aapl.get('margin_of_safety')}%")

if __name__ == "__main__":
    verify_performance()
