import sys
import json
import yfinance as yf
import pandas as pd
import traceback
import os
import tempfile

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

def fetch_data(symbols, start_date, end_date, interval, output_file):
    try:
        log(f"Starting fetch for {len(symbols)} symbols. Range: {start_date}-{end_date}, Int: {interval}")
        
        # Configure yf - Enable threads for speed since we are isolated
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
            prepost=False  # Explicitly disable pre/post market data
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
        
        symbols = request.get("symbols", [])
        start = request.get("start")
        end = request.get("end")
        interval = request.get("interval")
        output_file = request.get("output_file")
        
        if not output_file:
            # Generate a temp file path
            fd, output_file = tempfile.mkstemp(suffix=".json")
            os.close(fd)
        
        result = fetch_data(symbols, start, end, interval, output_file)
        
        # Print result metadata to stdout (path to file)
        print(json.dumps(result))
        
    except Exception as e:
        log(f"Worker main error: {e}")
        print(json.dumps({"status": "error", "message": f"Worker input error: {e}"}))
