import sys
import json
import traceback
import os
import tempfile
import logging # Added missing import
from datetime import datetime, date
import gc
import time
import random
import requests

# List of modern browser user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edge/119.0.0.0"
]

def get_requests_session():
    """Creates a requests session with a randomized User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    })
    return session

# Setup explicit logging
# Setup explicit logging
LOG_FILE = os.path.join(tempfile.gettempdir(), "worker_debug_investa.log")
def log(msg):
    try:
        logging.info(msg)
        # sys.stderr.write(f"WORKER: {msg}\n")
        # sys.stderr.flush()
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now()} - {msg}\n")
    except:
        pass

try:
    from market_fallback import fetch_data_fallback, fetch_info_fallback
except ImportError:
    log("market_fallback not found or failed to import. Redundancy disabled.")
    fetch_data_fallback = None
    fetch_info_fallback = None

# Removed ThreadPoolExecutor to save memory

def retry_with_backoff(retries=3, backoff_in_seconds=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limit indicators in the exception
                    err_str = str(e)
                    is_rate_limit = any(indicator in err_str for indicator in ["Too Many Requests", "429", "YFRateLimitError", "503", "Service Unavailable"])
                    
                    if x == retries or not is_rate_limit:
                        raise
                    
                    # More aggressive backoff for 429
                    base_delay = backoff_in_seconds if not is_rate_limit else backoff_in_seconds * 3
                    sleep = (base_delay * (2 ** x) + 
                             random.uniform(2, 8)) # More jitter
                    
                    log(f"Rate limited or Service error ({err_str[:50]}...). Retrying in {sleep:.2f} seconds... (Attempt {x+1}/{retries})")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def fetch_info(symbols, output_file, minimal=False):
    try:
        # Randomized jitter to prevent thundering herd
        if len(symbols) > 1:
            jitter = random.uniform(0.1, 1.5)
            log(f"Jittering for {jitter:.2f}s before batch info fetch...")
            time.sleep(jitter)

        log(f"Starting info fetch for {len(symbols)} symbols. Sequential mode. Minimal={minimal}")
        results = {}
        
        def get_single_info(sym):
            try:
                import yfinance as yf  # Redundant but safe
                log(f"Fetching .info for {sym}...")
                ticker = yf.Ticker(sym)
                @retry_with_backoff(retries=2, backoff_in_seconds=10)
                def _fetch_yf_ticker_info(t):
                    return t.info

                info = _fetch_yf_ticker_info(ticker)
                log(f"Received .info for {sym}. Keys: {len(info) if info else 0}")

                # --- ETF DATA EXTRACTION ---
                if not minimal:
                    try:
                        if hasattr(ticker, 'funds_data'):
                            log(f"Fetching funds_data for {sym}...")
                            fd = ticker.funds_data
                            if fd:
                                etf_data = {}
                                th = getattr(fd, 'top_holdings', None)
                                if th is not None:
                                    etf_data['top_holdings'] = []
                                    if hasattr(th, 'iterrows'): 
                                        for s, row in th.iterrows():
                                            etf_data['top_holdings'].append({
                                                "symbol": str(s), "name": str(row.get('Name', s)), "percent": float(row.get('Holding Percent', 0))
                                            })
                                if hasattr(fd, 'sector_weightings') and fd.sector_weightings:
                                    etf_data['sector_weightings'] = fd.sector_weightings
                                if hasattr(fd, 'asset_classes') and fd.asset_classes:
                                    etf_data['asset_classes'] = fd.asset_classes
                                if etf_data:
                                    info['etf_data'] = etf_data
                            log(f"funds_data processed for {sym}")
                    except Exception as e_etf:
                        log(f"Error extracting ETF data for {sym}: {e_etf}")
                    
                    # --- ANALYST ESTIMATES ---
                    try:
                        log(f"Fetching earnings_estimate for {sym}...")
                        ee = ticker.earnings_estimate
                        if not ee.empty:
                            info['_earnings_estimate'] = ee.to_dict(orient='index')
                        
                        log(f"Fetching revenue_estimate for {sym}...")
                        re = ticker.revenue_estimate
                        if not re.empty:
                            info['_revenue_estimate'] = re.to_dict(orient='index')
                        
                        log(f"Fetching growth_estimates for {sym}...")
                        ge = ticker.growth_estimates
                        if not ge.empty:
                            info['_growth_estimates'] = ge.to_dict(orient='index')
                        log(f"Analyst estimates processed for {sym}")
                    except Exception as e_analyst:
                        log(f"Error fetching analyst estimates for {sym}: {e_analyst}")
                else:
                    log(f"Minimal mode: Skipping ETF and Analyst data for {sym}")
                
                return sym, info
            except Exception as e:
                log(f"CRITICAL Error fetching {sym} via yf: {e}")
                if fetch_info_fallback:
                    log(f"Attempting fallback for {sym}...")
                    fb_info = fetch_info_fallback(sym)
                    if fb_info:
                        return sym, fb_info
                return sym, None

        # SEQUENTIAL: Removed ThreadPoolExecutor to minimize peak memory (prevent OOM)
        for i, sym in enumerate(symbols):
            log(f"Processing symbol {i+1}/{len(symbols)}: {sym}")
            sym, info = get_single_info(sym)
            results[sym] = info
            # Explicitly clear temporary objects and collect garbage after each symbol
            log(f"Collecting garbage after {sym}...")
            gc.collect()
            log(f"Memory cleared for {sym}")
        
        with open(output_file, "w") as f:
            json.dump({"status": "success", "data": results}, f)
        return {"status": "success", "file": output_file}
    except Exception as e:
        log(f"Error in fetch_info: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def fetch_statements_batch(symbols, period_type, output_file):
    try:
        import yfinance as yf # Lazy import
        log(f"Starting consolidated statements fetch ({period_type}) for {len(symbols)} symbols.")
        results = {}
        
        for i, sym in enumerate(symbols):
            log(f"Processing statements for {sym} ({i+1}/{len(symbols)})...")
            ticker = yf.Ticker(sym)
            sym_data = {}
            
            try:
                if period_type == "quarterly":
                    f = ticker.quarterly_financials
                    b = ticker.quarterly_balance_sheet
                    c = ticker.quarterly_cashflow
                else:
                    f = ticker.financials
                    b = ticker.balance_sheet
                    c = ticker.cashflow
                
                if f is not None and not f.empty:
                    sym_data['financials'] = f.to_json(orient='split', date_format='iso')
                if b is not None and not b.empty:
                    sym_data['balance_sheet'] = b.to_json(orient='split', date_format='iso')
                if c is not None and not c.empty:
                    sym_data['cashflow'] = c.to_json(orient='split', date_format='iso')
                
                results[sym] = sym_data
            except Exception as e_sym:
                log(f"Error fetching statements for {sym}: {e_sym}")
                results[sym] = None
            
            # Memory management
            log(f"Collecting garbage after {sym} statements...")
            gc.collect()
            
        with open(output_file, "w") as f:
            json.dump({"status": "success", "data": results}, f)
        return {"status": "success", "file": output_file}
        
    except Exception as e:
        log(f"Error in fetch_statements_batch: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def fetch_statement(symbol, statement_type, period_type, output_file):
    try:
        import yfinance as yf # Lazy import
        import pandas as pd  # Lazy import
        log(f"Starting statement fetch for {symbol}: {statement_type} ({period_type})")
        ticker = yf.Ticker(symbol)
        df = pd.DataFrame()
        if period_type == "quarterly":
            if statement_type == "financials": df = ticker.quarterly_financials
            elif statement_type == "balance_sheet": df = ticker.quarterly_balance_sheet
            elif statement_type == "cashflow": df = ticker.quarterly_cashflow
        else:
            if statement_type == "financials": df = ticker.financials
            elif statement_type == "balance_sheet": df = ticker.balance_sheet
            elif statement_type == "cashflow": df = ticker.cashflow
        
        if df is not None and not df.empty:
            json_str = df.to_json(orient='split', date_format='iso')
            with open(output_file, "w") as f: f.write(json_str)
            return {"status": "success", "file": output_file}
        else:
            return {"status": "success", "data": None}
    except Exception as e:
        log(f"Error in fetch_statement: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def fetch_dividends(symbol, output_file):
    try:
        import yfinance as yf # Lazy import
        log(f"Starting dividends fetch for {symbol}")
        ticker = yf.Ticker(symbol)
        divs = ticker.dividends
        if divs is not None and not divs.empty:
            # Convert Series to DataFrame to ensure consistent 'split' format
            df = divs.to_frame(name="Dividends")
            json_str = df.to_json(orient='split', date_format='iso')
            with open(output_file, "w") as f: f.write(json_str)
            return {"status": "success", "file": output_file}

        else:
            return {"status": "success", "data": None}
    except Exception as e:
        log(f"Error in fetch_dividends: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def fetch_calendar(symbol, output_file):
    try:
        import yfinance as yf # Lazy import
        import pandas as pd # Added missing import
        log(f"Starting calendar fetch for {symbol}")
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal:
            # Convert any datetime/date objects to strings for JSON
            processed_cal = {}
            for k, v in cal.items():
                if isinstance(v, (datetime, pd.Timestamp, date)):
                    processed_cal[k] = str(v)
                elif isinstance(v, list):
                    processed_cal[k] = [str(i) if isinstance(i, (datetime, pd.Timestamp, date)) else i for i in v]
                else:
                    processed_cal[k] = v
            
            with open(output_file, "w") as f:
                json.dump({"status": "success", "data": processed_cal}, f)
            return {"status": "success", "file": output_file}
        else:
            return {"status": "success", "data": None}
    except Exception as e:
        log(f"Error in fetch_calendar: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}



def fetch_data(symbols, start_date, end_date, interval, output_file, period=None):
    try:
        import yfinance as yf # Lazy import
        import pandas as pd  # Lazy import
        @retry_with_backoff(retries=3, backoff_in_seconds=10)
        def _execute_download(**kwargs):
            log(f"Executing yf.download attempt...")
            return yf.download(**kwargs)

        if period:
             log(f"Starting fetch for {len(symbols)} symbols. Period: {period}, Int: {interval}")
             data = _execute_download(
                tickers=symbols,
                period=period,
                interval=interval,
                progress=False,
                group_by="ticker",
                # Use unadjusted Close so historical portfolio valuation reflects
                # the real price the user paid/saw on each date. Adj Close (with
                # split+dividend back-adjustment) would systematically deflate
                # historical values for dividend-paying assets and create phantom
                # day-1 losses on every buy. Adj Close is still returned in a
                # separate column for benchmark/TWR consumers that need it.
                auto_adjust=False,
                actions=True,
                timeout=30,
                threads=1
            )
        else:
            log(f"Starting fetch for {len(symbols)} symbols. Range: {start_date}-{end_date}, Int: {interval}")
            # Randomized jitter for data fetch
            jitter = random.uniform(0.1, 2.0)
            time.sleep(jitter)
            
            data = _execute_download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                group_by="ticker",
                # Use unadjusted Close so historical portfolio valuation reflects
                # the real price the user paid/saw on each date. Adj Close (with
                # split+dividend back-adjustment) would systematically deflate
                # historical values for dividend-paying assets and create phantom
                # day-1 losses on every buy. Adj Close is still returned in a
                # separate column for benchmark/TWR consumers that need it.
                auto_adjust=False,
                actions=True,
                timeout=30,
                threads=1
            )
        
        # DIAGNOSTIC LOGGING
        log(f"Raw yf.download result: Shape={data.shape}, Columns={list(data.columns[:10])}")
        if not data.empty:
            log(f"Raw yf.download index range: {data.index[0]} to {data.index[-1]} (TZ: {data.index.tz})")
            for s in symbols[:2]: # Log first 2 symbols as sample
                 if isinstance(data.columns, pd.MultiIndex):
                     # Check if the symbol exists in the MultiIndex at level 1
                     if s in data.columns.get_level_values(1):
                         s_data = data.xs(s, axis=1, level=1, drop_level=False) # Keep level for consistency in shape check
                     else:
                         s_data = pd.DataFrame() # Symbol not found
                 else:
                     s_data = data[s] if s in data.columns else pd.DataFrame()
                 log(f"  Symbol {s}: {len(s_data)} rows. Tuesday rows: {len(s_data[s_data.index.date == date(2026, 1, 20)]) if not s_data.empty else 0}")
        
        # RATE LIMIT DETECTION
        if "Too Many Requests" in str(data) or "YFRateLimitError" in str(data):
            log("YFinance Rate Limit Detected in download result.")
        
        # NOTE: We previously attempted manual filtering (9:30-16:00) but it caused issues
        # with UTC vs Local timezones (pruning valid data).
        # Since 'prepost=False' returns exactly 78 rows (6.5 hours) for 5m interval,
        # it is properly respecting market hours. We rely on yfinance for this.
        
        if not data.empty:
            log(f"Data index dtype: {data.index.dtype}")
            try:
                # Normalize timezone to US/Eastern for consistent date masking (9:30-16:00 filtering uses NY)
                # This handles both UTC-aware and naive data.
                # IMPORTANT: We build a TZ-aware copy (df_est) for filtering logic, but
                # we apply the resulting boolean mask positionally to 'data' (original TZ)
                # to avoid index-alignment mismatches between different TZ-aware indexes.
                df_est = data.copy()
                if df_est.index.tz is None:
                    # Assume naive is UTC for yfinance (usually is)
                    df_est.index = df_est.index.tz_localize("UTC").tz_convert("America/New_York")
                else:
                    df_est.index = df_est.index.tz_convert("America/New_York")
                
                # Apply localized index for filtering
                original_len = len(df_est)
                
                # 1. Intra-day Time Filter (only for short intervals)
                if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                    # Filter for market hours (09:30 - 16:15) using NY-localized index
                    df_filtered_est = df_est.between_time("09:30", "16:15")
                    if not df_filtered_est.empty:
                        # Build a positional boolean mask (same length as df_est/data)
                        # to avoid tz_convert(None) issues when data.index.tz is naive
                        time_mask = df_est.index.isin(df_filtered_est.index)
                        data = data.loc[time_mask]  # Apply positionally - safe as lengths match
                        df_est = df_est.loc[time_mask]  # Keep df_est in sync
                        log(f"Time-Filtered (NY 09:30-16:15): {original_len} -> {len(data)}.")
                    else:
                        data = pd.DataFrame()  # Became empty
                        log(f"Time-Filtered (NY 09:30-16:15): {original_len} -> 0 (EMPTY)")
                
                # 2. Date Mask Filter (for all intervals)
                # If start_date/end_date provided, enforce them using the EST localized index.
                # FIX: We use df_est (already NY-converted, same length as data) to build the
                # mask, then apply it positionally to 'data'. This avoids any TZ mismatch.
                if not data.empty and start_date and end_date:
                    # df_est is already in sync with data (same positional rows) from above.
                    # If data was modified by time filter, df_est was also updated above.
                    # Safety: rebuild df_est from current data if lengths differ
                    if len(df_est) != len(data):
                        df_est = data.copy()
                        if df_est.index.tz is None:
                            df_est.index = df_est.index.tz_localize("UTC").tz_convert("America/New_York")
                        else:
                            df_est.index = df_est.index.tz_convert("America/New_York")
                    
                    start_d = pd.to_datetime(start_date).date()
                    end_d = pd.to_datetime(end_date).date()
                    date_mask = (df_est.index.date >= start_d) & (df_est.index.date <= end_d)
                    data = data.loc[date_mask]  # Apply positionally - safe as lengths match
                    log(f"Date-Filtered ({start_date} to {end_date}): New length {len(data)}")

            except Exception as e:
                log(f"Timezone filtering error: {e}")
                # Fallback: don't filter if error
                pass
            
            # DEDUPLICATION: Ensure unique timestamps before returning
            # Duplicate timestamps can cause a 500 error in alignment logic (reindex)
            if not data.empty:
                original_len = len(data)
                data = data[~data.index.duplicated(keep='last')]
                if len(data) < original_len:
                    log(f"Deduplicated index: {original_len} -> {len(data)}")
        log(f"Fetch completed. Data shape: {data.shape}")
        
        if data.empty:
            log(f"Data is empty from yfinance. (Tickers: {symbols[:5]}, Range: {start_date}-{end_date if end_date else 'N/A'}, Interval: {interval})")
            
            # Diagnostic: check if it might be a holiday or weekend if we have a specific date
            fallback_success = False
            if fetch_data_fallback and len(symbols) == 1:
                # Basic fallback implementation handles single symbols best
                log(f"Attempting fallback for {symbols[0]}...")
                fb_data = fetch_data_fallback(symbols, start_date, end_date, interval)
                if fb_data is not None and not fb_data.empty:
                    data = fb_data
                    fallback_success = True
                    log(f"Fallback SUCCESS for {symbols[0]}. Shape: {data.shape}")
                    
            if not fallback_success:
                return {"status": "success", "data": None, "message": "No data returned (likely closed market or invalid range)"}

        # Save to temp file using parquet or json to avoid massive string in memory
        # Parquet is efficient but requires pyarrow/fastparquet. JSON is safer for compat.
        # json_orient="split"
        
        json_str = data.to_json(orient='split', date_format='iso')
        log(f"Data serialized. Length: {len(json_str)}")
        
        with open(output_file, "w") as f:
            f.write(json_str)
            
        return {"status": "success", "file": output_file}

    except Exception as e:
        log(f"Error in fetch_data: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    sys.stderr.write("WORKER MAIN STARTING...\n")
    try:
        # Read input from stdin
        input_json = sys.stdin.read()
        request = json.loads(input_json)
        
        task = request.get("task", "history")
        symbols = request.get("symbols", [])
        start = request.get("start")
        end = request.get("end")
        period = request.get("period")
        interval = request.get("interval", "1d")
        output_file = request.get("output_file")
        
        if not output_file:
            # Generate a temp file path
            fd, output_file = tempfile.mkstemp(suffix=".json")
            os.close(fd)
        
        if task == "info":
            result = fetch_info(symbols, output_file, minimal=request.get("minimal", False))
        elif task == "statement":
            result = fetch_statement(symbols[0], request.get("statement_type"), request.get("period_type"), output_file)
        elif task == "statements_batch":
            result = fetch_statements_batch(symbols, request.get("period_type"), output_file)
        elif task == "dividends":
            result = fetch_dividends(symbols[0], output_file)
        elif task == "calendar":
            result = fetch_calendar(symbols[0], output_file)
        else:
            result = fetch_data(symbols, start, end, interval, output_file, period=period)


        
        # Print result metadata to stdout (path to file)
        print(json.dumps(result))
        
    except Exception as e:
        log(f"Worker main error: {e}")
        print(json.dumps({"status": "error", "message": f"Worker input error: {e}"}))
