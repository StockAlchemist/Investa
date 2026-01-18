import yfinance as yf
import json

def inspect_etf(symbol):
    ticker = yf.Ticker(symbol)
    
    print(f"--- Data for {symbol} ---")
    
    # Check info
    info = ticker.info
    keys_of_interest = [
        "quoteType", "longName", "expenseRatio", "yield", "navPrice", 
        "category", "fundFamily", "netAssets", "legalType"
    ]
    

    
    info_subset = {k: info.get(k) for k in keys_of_interest}
    print("Info Subset:")
    print(json.dumps(info_subset, indent=2))
    
    # Check holdings if available (funds_data)
    # yfinance often puts fund holdings in funds_data (which might be private/internal?)
    # or .holdings or .fund_holding_info
    
    # Let's check typical attributes
    try:
        if hasattr(ticker, 'funds_data'):
            fd = ticker.funds_data
            if fd:
                print("--- Funds Data Attributes ---")
                try:
                    if hasattr(fd, 'top_holdings'):
                        print("Top Holdings:", fd.top_holdings)
                    if hasattr(fd, 'sector_weightings'):
                        print("Sector Weightings:", fd.sector_weightings)
                    if hasattr(fd, 'asset_classes'):
                        print("Asset Classes:", fd.asset_classes)
                except Exception as e:
                    print(f"Error accessing funds_data attributes: {e}")
    except Exception as e:
        print(f"Funds Data Error: {e}")
        
    # Check if there is 'topHoldings' in info
    if 'holdings' in info: # Unlikely, but let's check
        print("Info 'holdings':", info['holdings'])

if __name__ == "__main__":
    inspect_etf("SCHG")
