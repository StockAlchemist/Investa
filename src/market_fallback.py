import os
import requests
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Fallback: Alpha Vantage (requires ALPHA_VANTAGE_API_KEY environment variable)
# Alternatively, could use Finnhub or another free API if Alpha Vantage limits are hit.
AV_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

def fetch_data_fallback(symbols: list, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
    """
    Attempts to fetch historical data from a fallback provider.
    Currently implements a simple Alpha Vantage daily mock/fetch for a single symbol.
    In a complete implementation, this would handle batching and interval mapping.
    """
    if not symbols:
        return None
        
    symbol = symbols[0] # Focus on primary symbol for fallback simplicity
    logger.warning(f"Using fallback provider (Alpha Vantage) for {symbol}...")
    
    # Map intervals (yf to Alpha Vantage)
    av_interval_map = {
        "1d": "TIME_SERIES_DAILY",
        "1wk": "TIME_SERIES_WEEKLY",
        "1mo": "TIME_SERIES_MONTHLY",
        "1m": "TIME_SERIES_INTRADAY",
        "5m": "TIME_SERIES_INTRADAY",
        "15m": "TIME_SERIES_INTRADAY",
        "30m": "TIME_SERIES_INTRADAY",
        "60m": "TIME_SERIES_INTRADAY",
    }
    
    function = av_interval_map.get(interval, "TIME_SERIES_DAILY")
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={AV_API_KEY}"
    
    if "INTRADAY" in function:
        # Alpha Vantage uses '1min', '5min', '15min', '30min', '60min'
        av_int = interval.replace('m', 'min').replace('60min', '60min') # Quick map
        url += f"&interval={av_int}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for Alpha Vantage error or rate limit
        if "Error Message" in data or "Note" in data:
            logger.error(f"Alpha Vantage Fallback Error/Rate Limit: {data}")
            return None
            
        # Parse the specific time series key
        ts_key = next((key for key in data.keys() if "Time Series" in key), None)
        if not ts_key:
            return None
            
        df = pd.DataFrame.from_dict(data[ts_key], orient="index")
        df.index = pd.to_datetime(df.index)
        
        # Rename columns to match yfinance format (Open, High, Low, Close, Volume)
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
            "5. adjusted close": "Adj Close" # Might be present in daily adjusted
        }, inplace=True, errors="ignore")
        
        # Convert types to float
        df = df.astype(float)
        
        # Create MultiIndex format if multiple symbols were expected by the caller
        # to loosely match yfinance group_by="ticker" format if needed,
        # but the worker normally expects straightforward DataFrame for single symbol or MultiIndex
        # For simplicity in fallback, we return a flat dataframe for the symbol
        
        # Filter by start/end if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        return df

    except Exception as e:
        logger.error(f"Fallback fetch failed for {symbol}: {e}")
        return None

def fetch_info_fallback(symbol: str) -> Optional[Dict]:
    """
    Attempts to fetch basic info/quotes from a fallback provider (Finnhub if available).
    """
    if FINNHUB_API_KEY:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data and "c" in data: # Current price exists
                return {
                    "symbol": symbol,
                    "currentPrice": data.get("c"),
                    "previousClose": data.get("pc"),
                    "open": data.get("o"),
                    "dayHigh": data.get("h"),
                    "dayLow": data.get("l"),
                    "_fallback_used": True
                }
        except Exception as e:
            logger.error(f"Finnhub fallback failed for {symbol}: {e}")
            
    return None
