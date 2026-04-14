import sys
import os
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/My Drive/Finance/Investa"
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# Mock config
import config
# Ensure environment has a dummy key just for the prompt test
os.environ["GEMINI_API_KEY"] = "dummy"

from server.ai_analyzer import generate_stock_review

# Mock data
symbol = "AAPL"
fund_data = {"marketCap": 3000000000000}
ratios = {"Return on Equity (ROE) (%)": 100}

print("Testing generate_stock_review prompt construction...")
# We don't actually need to hit the API, just need to see if the f-string works
# Wait, generate_stock_review actually calls the API. 
# I can just monkeypatch the request or just check if it reaches the request part.

import requests
from unittest.mock import MagicMock

# Mock requests.post to avoid hitting the API
requests.post = MagicMock()

try:
    generate_stock_review(symbol, fund_data, ratios, force_refresh=True)
    print("Prompt construction successful!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
