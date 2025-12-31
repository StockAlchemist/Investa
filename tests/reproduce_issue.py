
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock config if needed, but it seems finutils handles missing config somewhat.
# However, to be safe, let's just try to import.

from finutils import map_to_yf_symbol, is_cash_symbol

# Setup basic logging to stdout to see if warnings appear
logging.basicConfig(level=logging.DEBUG)

def test_cash_mapping():
    print("Testing 'Cash ($)' mapping...")
    res = map_to_yf_symbol("Cash ($)", {}, set())
    if res is None:
        print("PASS: 'Cash ($)' mapped to None")
    else:
        print(f"FAIL: 'Cash ($)' mapped to {res}")

    print("\nTesting 'Cash (฿)' mapping...")
    res = map_to_yf_symbol("Cash (฿)", {}, set())
    if res is None:
        print("PASS: 'Cash (฿)' mapped to None")
    else:
        print(f"FAIL: 'Cash (฿)' mapped to {res}")

    print("\nTesting 'AAPL' mapping...")
    res = map_to_yf_symbol("AAPL", {}, set())
    if res == "AAPL":
        print("PASS: 'AAPL' mapped to AAPL")
    else:
        print(f"FAIL: 'AAPL' mapped to {res}")

if __name__ == "__main__":
    test_cash_mapping()
