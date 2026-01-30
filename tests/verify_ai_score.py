import os
import sys
import json
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "server")))

from screener_service import screen_stocks
from db_utils import get_db_connection

logging.basicConfig(level=logging.INFO)

def test_ai_score(target_symbol):
    print(f"Testing AI Score for {target_symbol}...")
    # Call screen_stocks for target_symbol only
    results = screen_stocks(universe_type="manual", manual_symbols=[target_symbol])
    
    if not results:
        print("FAIL: No results returned")
        return

    stock = next((r for r in results if r['symbol'] == target_symbol), None)
    if not stock:
        print(f"FAIL: {target_symbol} not found in results")
        return

    print(f"Symbol: {stock['symbol']}")
    print(f"Name: {stock['name']}")
    print(f"Price: {stock['price']}")
    print(f"Intrinsic Value: {stock['intrinsic_value']}")
    print(f"Has AI Review: {stock['has_ai_review']}")
    print(f"AI Score: {stock['ai_score']}")
    if stock.get('ai_summary'):
        print(f"AI Summary: {stock['ai_summary'][:100]}...")
    else:
        print("AI Summary: None")

    # Assertions
    assert stock['has_ai_review'] is True, "has_ai_review should be True"
    assert stock['ai_score'] is not None, "ai_score should not be None"
    
    print(f"\nSUCCESS: AI data for {target_symbol} is correctly populated!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_ai_score(sys.argv[1].upper())
    else:
        test_ai_score("SYK")
        test_ai_score("SWKS")
