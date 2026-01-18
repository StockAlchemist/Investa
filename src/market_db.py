import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
import threading
import config

class MarketDatabase:
    """
    Manages a persistent SQLite database for historical market data.
    Provides methods for upserting, querying, and checking data integrity.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(config.get_app_data_dir(), "market_data.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        # Use thread-local storage for DB connections
        if not hasattr(self, '_local_storage'):
            self._local_storage = threading.local()
        
        if not hasattr(self._local_storage, 'conn'):
            import time
            retries = 5
            last_error = None
            for i in range(retries):
                try:
                    # check_same_thread=False is needed if the connection is passed between threads,
                    # but here we use thread-local so it should be fine with default True.
                    # However, timeout=30 is good for concurrent writes.
                    self._local_storage.conn = sqlite3.connect(self.db_path, timeout=30)
                    # Enable WAL mode for better concurrency
                    self._local_storage.conn.execute("PRAGMA journal_mode=WAL;")
                    self._local_storage.conn.execute("PRAGMA synchronous=NORMAL;")
                    return self._local_storage.conn
                except sqlite3.OperationalError as e:
                    last_error = e
                    if "unable to open database file" in str(e) or "database is locked" in str(e):
                        if i < retries - 1:
                            time.sleep(0.1 * (i + 1))
                            continue
                    raise e
            if last_error:
                raise last_error
        
        return self._local_storage.conn

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Historical OHLCV Table (Daily)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_ohlcv (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    interval TEXT DEFAULT '1d',
                    PRIMARY KEY (symbol, date, interval)
                )
            """)

            # Historical FX Table (Daily)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_fx (
                    pair TEXT,
                    date TEXT,
                    rate REAL,
                    interval TEXT DEFAULT '1d',
                    PRIMARY KEY (pair, date, interval)
                )
            """)

            # Sync Metadata Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_synced TEXT,
                    inception_date TEXT,
                    info_json TEXT
                )
            """)

            # Intraday OHLCV Table (High Frequency)
            # Uses timestamp (ISO with time) instead of date
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intraday_ohlcv (
                    symbol TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    interval TEXT,
                    PRIMARY KEY (symbol, timestamp, interval)
                )
            """)
            conn.commit()

    def upsert_ohlcv(self, symbol: str, df: pd.DataFrame, interval: str = '1d'):
        """
        Upserts OHLCV data from a DataFrame.
        DataFrame must have a DatetimeIndex.
        """
        if df.empty:
            return


        with self._get_connection() as conn:
            cursor = conn.cursor()
            for timestamp, row in df.iterrows():
                # Ensure date is string YYYY-MM-DD
                date_str = timestamp.strftime('%Y-%m-%d') if hasattr(timestamp, 'strftime') else str(timestamp)[:10]
                
                # Helper to clean NaNs
                def clean(val):
                    if pd.isna(val): return None
                    return float(val)

                # Normalize columns
                open_val = clean(row.get('Open'))
                high_val = clean(row.get('High'))
                low_val = clean(row.get('Low'))
                close_val = clean(row.get('Close', row.get('price')))
                adj_close_val = clean(row.get('Adj Close', close_val))
                volume_val = int(row.get('Volume', 0)) if pd.notna(row.get('Volume', 0)) else 0

                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO daily_ohlcv 
                        (symbol, date, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, 
                        date_str, 
                        open_val, 
                        high_val, 
                        low_val, 
                        close_val, 
                        adj_close_val, 
                        volume_val,
                        interval
                    ))
                except Exception as e_ins:
                    logging.error(f"DB Upsert Error for {symbol} on {date_str}: {e_ins}")
            
            conn.commit()
            
            # Update sync metadata
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO sync_metadata (symbol, last_synced) VALUES (?, ?)
            """, (symbol, now))
            conn.commit()

    def upsert_fx(self, pair: str, df: pd.DataFrame, interval: str = '1d'):
        """
        Upserts FX rate data from a DataFrame.
        DataFrame must have a DatetimeIndex and a column 'Close' or 'rate'.
        """
        if df.empty:
            return

        col = 'Close' if 'Close' in df.columns else (df.columns[0] if not df.empty else None)
        if not col:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for timestamp, row in df.iterrows():
                date_str = timestamp.strftime('%Y-%m-%d') if hasattr(timestamp, 'strftime') else str(timestamp)[:10]
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_fx 
                    (pair, date, rate, interval)
                    VALUES (?, ?, ?, ?)
                """, (pair, date_str, row[col], interval))
            conn.commit()

    def get_ohlcv(self, symbol: str, start_date: date, end_date: date, interval: str = '1d') -> pd.DataFrame:
        """Retrieves OHLCV data for a symbol within a date range."""
        query = """
            SELECT date, open, high, low, close, adj_close, volume 
            FROM daily_ohlcv 
            WHERE symbol = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval, str(start_date), str(end_date)))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # Rename columns to standard YF format for compatibility
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # FORCE NUMERIC TYPES
            # This is critical because if 'Open' contains None (which sqlite returns for NULL), 
            # pandas treats the column as 'object', causing interpolation to fail/warn.
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        return df

    def get_ohlcv_batch(self, symbols: List[str], start_date: date, end_date: date, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Retrieves OHLCV data for multiple symbols in a collection of DataFrames."""
        if not symbols:
            return {}
            
        placeholders = ', '.join(['?'] * len(symbols))
        query = f"""
            SELECT symbol, date, open, high, low, close, adj_close, volume 
            FROM daily_ohlcv 
            WHERE symbol IN ({placeholders}) AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY symbol, date ASC
        """
        
        results = {}
        with self._get_connection() as conn:
            # We must pass the list of symbols first, then other params
            params = symbols + [interval, str(start_date), str(end_date)]
            df_all = pd.read_sql_query(query, conn, params=params)
        
        if not df_all.empty:
            df_all['date'] = pd.to_datetime(df_all['date'])
            # Group by symbol and process each
            for sym, group in df_all.groupby('symbol'):
                df = group.drop(columns=['symbol'])
                df.set_index('date', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # FORCE NUMERIC
                cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                for c in cols_to_numeric:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                results[sym] = df
                
        return results

    def upsert_intraday(self, symbol: str, df: pd.DataFrame, interval: str):
        """
        Upserts Intraday OHLCV data from a DataFrame.
        DataFrame must have a DatetimeIndex.
        Dates are stored as ISO timestamps (YYYY-MM-DDTHH:MM:SS...).
        """
        if df.empty:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for timestamp, row in df.iterrows():
                # Store full timestamp ISO string
                ts_str = timestamp.isoformat() 
                
                # Helper to clean NaNs
                def clean(val):
                    if pd.isna(val): return None
                    return float(val)

                # Normalize columns
                open_val = clean(row.get('Open'))
                high_val = clean(row.get('High'))
                low_val = clean(row.get('Low'))
                close_val = clean(row.get('Close', row.get('price')))
                # Intraday usually doesn't have Adj Close diffs, but store if present
                adj_close_val = clean(row.get('Adj Close', close_val))
                volume_val = int(row.get('Volume', 0)) if pd.notna(row.get('Volume', 0)) else 0

                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO intraday_ohlcv 
                        (symbol, timestamp, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, 
                        ts_str, 
                        open_val, 
                        high_val, 
                        low_val, 
                        close_val, 
                        adj_close_val, 
                        volume_val,
                        interval
                    ))
                except Exception as e_ins:
                    logging.error(f"DB Intraday Upsert Error for {symbol} on {ts_str}: {e_ins}")
            
            conn.commit()

    def get_intraday(self, symbol: str, start_ts: datetime, end_ts: datetime, interval: str) -> pd.DataFrame:
        """Retrieves Intraday OHLCV data for a symbol within a timestamp range."""
        query = """
            SELECT timestamp, open, high, low, close, adj_close, volume 
            FROM intraday_ohlcv 
            WHERE symbol = ? AND interval = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        # Convert search bounds to ISO strings for comparison
        start_str = start_ts.isoformat()
        end_str = end_ts.isoformat()
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval, start_str, end_str))
        
        if not df.empty:
            # Fix for parsing mixed format ISO strings with timezone info
            # "YYYY-MM-DDTHH:MM:SS+00:00" might fail with default parser in some pandas versions
            # Enforce utc=True to avoid mixed naive/aware comparisons and FutureWarnings
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = "Date" # Standardize name for portfolio_logic compatibility
            
            # Rename columns to standard YF format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # FORCE NUMERIC TYPES
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        return df

    def get_fx(self, pair: str, start_date: date, end_date: date, interval: str = '1d') -> pd.DataFrame:
        """Retrieves FX rate data for a pair within a date range."""
        query = """
            SELECT date, rate 
            FROM daily_fx 
            WHERE pair = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(pair, interval, str(start_date), str(end_date)))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.columns = ['price'] # Map to 'price' which is expected by portfolio_logic
        return df

    def get_last_date(self, symbol: str, table: str = "daily_ohlcv") -> Optional[date]:
        """Returns the most recent date available in the DB for a symbol."""
        col = "symbol" if table == "daily_ohlcv" else "pair"
        query = f"SELECT MAX(date) FROM {table} WHERE {col} = ?"
        with self._get_connection() as conn:
            res = conn.execute(query, (symbol,)).fetchone()
            if res and res[0]:
                return datetime.strptime(res[0], '%Y-%m-%d').date()
        return None

    def get_last_dates(self, symbols: List[str], table: str = "daily_ohlcv") -> Dict[str, date]:
        """Returns a dict of {symbol: last_date} for a list of symbols."""
        col = "symbol" if table == "daily_ohlcv" else "pair"
        placeholders = ', '.join(['?'] * len(symbols))
        query = f"SELECT {col}, MAX(date) FROM {table} WHERE {col} IN ({placeholders}) GROUP BY {col}"
        
        results = {}
        with self._get_connection() as conn:
            cursor = conn.execute(query, symbols)
            for row in cursor:
                if row[1]:
                    results[row[0]] = datetime.strptime(row[1], '%Y-%m-%d').date()
        return results

    def get_sync_metadata_batch(self, symbols: List[str]) -> Dict[str, datetime]:
        """Returns a dict of {symbol: last_synced_datetime} for a list of symbols."""
        placeholders = ', '.join(['?'] * len(symbols))
        query = f"SELECT symbol, last_synced FROM sync_metadata WHERE symbol IN ({placeholders})"
        
        results = {}
        with self._get_connection() as conn:
            cursor = conn.execute(query, symbols)
            for row in cursor:
                if row[1]:
                    try:
                        results[row[0]] = datetime.fromisoformat(row[1])
                    except ValueError:
                        pass
        return results

    def check_integrity(self, symbol: str, new_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Compares new_df with existing DB data for overlapping dates.
        Returns (is_consistent, reason).
        If inconsistencies are found (e.g., adj_close differed significantly), 
        it suggests a re-fetch of history.
        """
        if new_df.empty:
            return True, None

        overlap_dates = new_df.index.strftime('%Y-%m-%d').tolist()
        placeholders = ', '.join(['?'] * len(overlap_dates))
        
        query = f"""
            SELECT date, adj_close 
            FROM daily_ohlcv 
            WHERE symbol = ? AND date IN ({placeholders})
        """
        
        with self._get_connection() as conn:
            db_data = pd.read_sql_query(query, conn, params=[symbol] + overlap_dates)
        
        if db_data.empty:
            return True, None # No overlap, no conflict

        db_data.set_index('date', inplace=True)
        
        for idx, row in new_df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            if date_str in db_data.index:
                db_val = db_data.loc[date_str, 'adj_close']
                new_val = row.get('Adj Close', row.get('Close'))
                
                if db_val and new_val:
                    # Check for significant diff (e.g. > 0.1%)
                    # This could be a split or dividend adjustment
                    diff = abs(db_val - new_val) / db_val
                    if diff > 0.001:
                        return False, f"Inconsistency detected on {date_str}: DB={db_val}, New={new_val} (diff={diff:.4%})"
        
        return True, None
