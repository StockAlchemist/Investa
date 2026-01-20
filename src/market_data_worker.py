import sys
import json
import yfinance as yf
import pandas as pd
import traceback
import os
import tempfile
from datetime import datetime, date

# Setup explicit logging
# Setup explicit logging
# Use temp path to force visibility
import tempfile
LOG_FILE = os.path.join(tempfile.gettempdir(), "worker_debug_investa.log")
def log(msg):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{pd.Timestamp.now()} - {msg}\n")
    except:
        pass

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_info(symbols, output_file):
    try:
        log(f"Starting info fetch for {len(symbols)} symbols with parallel threads.")
        results = {}
        
        def get_single_info(sym):
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info
                # --- ETF DATA EXTRACTION ---
                try:
                    if hasattr(ticker, 'funds_data'):
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
                except Exception as e_etf:
                    log(f"Error extracting ETF data for {sym}: {e_etf}")
                return sym, info
            except Exception as e:
                log(f"Error fetching info for {sym}: {e}")
                return sym, None

        # Use 10 threads for info fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(get_single_info, sym): sym for sym in symbols}
            for future in as_completed(future_to_sym):
                sym, info = future.result()
                results[sym] = info
        
        with open(output_file, "w") as f:
            json.dump({"status": "success", "data": results}, f)
        return {"status": "success", "file": output_file}
    except Exception as e:
        log(f"Error in fetch_info: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def fetch_statement(symbol, statement_type, period_type, output_file):
    try:
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
        if period:
             log(f"Starting fetch for {len(symbols)} symbols. Period: {period}, Int: {interval}")
             data = yf.download(
                tickers=symbols,
                period=period,
                interval=interval,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                actions=False,
                timeout=30,
                threads=False
            )
        else:
            log(f"Starting fetch for {len(symbols)} symbols. Range: {start_date}-{end_date}, Int: {interval}")
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                actions=False,
                timeout=30,
                threads=False
            )
        
        # NOTE: We previously attempted manual filtering (9:30-16:00) but it caused issues
        # with UTC vs Local timezones (pruning valid data).
        # Since 'prepost=False' returns exactly 78 rows (6.5 hours) for 5m interval,
        # it is properly respecting market hours. We rely on yfinance for this.
        
        if not data.empty:
            log(f"Data index dtype: {data.index.dtype}")
            try:
                # Normalize timezone to US/Eastern for consistent 9:30-16:00 filtering
                # This handles both UTC-aware and naive (assumed NY) data
                if data.index.tz is None:
                     # Assume NY if naive (common for yf)
                     data_local = data.index.tz_localize("America/New_York", ambiguous='infer')
                else:
                     # Convert to NY if aware
                     data_local = data.index.tz_convert("America/New_York")
                
                # Apply filter on the localized index
                # We need to reconstruct the dataframe with the filtered index, or use boolean mask
                # using between_time on the Frame directly is easiest, but strict on index type
                
                # Create a copy with the local index to filter
                df_local = data.copy()
                df_local.index = data_local
                
                original_len = len(df_local)
                # FILTERING: Only for intraday
                if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                    df_filtered_local = df_local.between_time("09:30", "16:00")
                    # Convert back to UTC
                    if not df_filtered_local.empty:
                        data = df_filtered_local.tz_convert("UTC")
                        log(f"Filtered (NY time 9:30-16:00): {original_len} -> {len(data)}")
                        log(f"Filtered Start (NY): {df_filtered_local.index[0]}")
                        log(f"Filtered End (NY): {df_filtered_local.index[-1]}")
                    else:
                        data = df_filtered_local # Still empty but avoids log errors
                        log(f"Filtered (NY time 9:30-16:00): {original_len} -> 0 (EMPTY)")
                else:
                     # For Daily/Weekly/Monthly, DO NOT FILTER TIME.
                     # Timestamps are often 00:00 UTC, which would be filtered out.
                     data = data # Keep original
                     log(f"Skipping time filter for interval {interval}")
                
            except Exception as e:
                log(f"Timezone filtering error: {e}")
                # Fallback: don't filter if error
                pass
                 
        log(f"Fetch completed. Data shape: {data.shape}")
        
        if data.empty:
            log("Data is empty.")
            return {"status": "success", "data": None}

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
            result = fetch_info(symbols, output_file)
        elif task == "statement":
            result = fetch_statement(symbols[0], request.get("statement_type"), request.get("period_type"), output_file)
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
