import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
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
        return sqlite3.connect(self.db_path)

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
                
                # Normalize columns
                open_val = row.get('Open')
                high_val = row.get('High')
                low_val = row.get('Low')
                close_val = row.get('Close', row.get('price'))
                adj_close_val = row.get('Adj Close', close_val)
                volume_val = row.get('Volume', 0)

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
