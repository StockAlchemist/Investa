import os
import sys
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "server")))

from screener_service import screen_stocks

logging.basicConfig(level=logging.INFO)

def test_batch_screener():
    print("Testing Batch Screener (Stress Test)...")
    # Take a few symbols from SP500 or just manual list
    symbols = ["SYK", "SWKS", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B"]
    
    try:
        results = screen_stocks(universe_type="manual", manual_symbols=symbols)
        print(f"SUCCESS: Returned {len(results)} results")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_screener()
