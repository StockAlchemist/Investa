import sys
import os
import time
import logging

# Add src to path so we can import modules
sys.path.append(os.path.join(os.getcwd(), "src"))

from server.screener_service import screen_stocks
from config import get_app_data_dir

# Setup specific logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def benchmark(universe_type):
    print(f"\n--- Benchmarking {universe_type} ---")
    start_time = time.time()
    try:
        # Run with fast_mode=False to force processing
        # This will fetch quotes and calculate IV
        results = screen_stocks(universe_type=universe_type, fast_mode=False)
        duration = time.time() - start_time
        print(f"Completed {universe_type} in {duration:.2f} seconds.")
        print(f"Results count: {len(results)}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test sp400 since that was in the screenshot
    benchmark("sp400")
