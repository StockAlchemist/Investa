import requests
import sys

def test_market_status():
    url = "http://localhost:8000/api/market_status"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Market Status: {data}")
        if "is_open" in data and isinstance(data["is_open"], bool):
            print("Verification SUCCESS: /api/market_status returned a valid boolean.")
        else:
            print("Verification FAILED: /api/market_status returned invalid data format.")
            sys.exit(1)
    except Exception as e:
        print(f"Verification FAILED: Could not connect to API: {e}")
        # Note: If the server is not running, this will fail. 
        # But I can't start the server if it's already running or if it needs specific env vars.
        # I'll check if the server is running first.
        sys.exit(1)

if __name__ == "__main__":
    test_market_status()
