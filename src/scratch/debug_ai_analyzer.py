import sys
import os
import json
import logging

# Add src to path
project_root = "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/My Drive/Finance/Investa"
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# Mock some things before importing ai_analyzer
import types
mock_dotenv = types.ModuleType('dotenv')
mock_dotenv.load_dotenv = lambda: None
sys.modules['dotenv'] = mock_dotenv

# Set logging to see failures
logging.basicConfig(level=logging.INFO)

try:
    from server.ai_analyzer import generate_stock_review
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Mock data
symbol = "AAPL"
fund_data = {"marketCap": 3000000000000}
ratios = {"Return on Equity (ROE) (%)": 100}

print(f"Testing generate_stock_review for {symbol}...")

# Set a dummy key if it doesn't exist just to test prompt construction
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "DUMMY_KEY"

try:
    # We want to catch errors in prompt construction or the request logic
    # To test prompt construction we can just check if we can get past the f-string
    # To test the whole thing we'd need a real key, but we want to see if it crashes before the request
    
    import requests
    from unittest.mock import MagicMock
    
    # Mock requests.post so we don't actually hit the API
    original_post = requests.post
    requests.post = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}))
    
    result = generate_stock_review(symbol, fund_data, ratios, force_refresh=True)
    print("Success! Result keys:", result.keys())
    
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore mock if necessary
    pass
