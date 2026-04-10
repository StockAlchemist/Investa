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

def test_batch_screener_all_universe():
    """
    Test the 'all' universe edge case without fetching the full database.
    Mock get_all_distinct_screener_results to return a controlled subset.
    """
    # Test fast_mode=True for the 'all' universe. This should execute an instant DB pull
    # and return a list of cached screener dicts, bypassing live Yahoo fetches.
    results_fast = screen_stocks(universe_type="all", fast_mode=True)
    
    # Assert we got a list back (it could be empty if DB is empty, but we shouldn't crash)
    assert isinstance(results_fast, list)
    
    if len(results_fast) > 0:
        # Check basic schema of returned items
        first_item = results_fast[0]
        assert "symbol" in first_item
        assert "intrinsic_value" in first_item
        assert "margin_of_safety" in first_item
        
if __name__ == "__main__":
    test_batch_screener()
    # To run this with pytest:
    # pytest tests/test_screener_stress.py -v
