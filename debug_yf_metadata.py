import yfinance as yf
import json

def test_metadata():
    symbol = "AAPL"
    print(f"Fetching metadata for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        print("\n--- Metadata Keys & Values (Subset) ---")
        keys_to_check = ["exchange", "fullExchangeName", "quoteType", "market", "currency", "shortName", "longName"]
        for k in keys_to_check:
            print(f"{k}: {info.get(k)}")
            
        print("\n--- All Keys containing 'exchange' ---")
        for k in info.keys():
            if "exchange" in k.lower():
                print(f"{k}: {info[k]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_metadata()
