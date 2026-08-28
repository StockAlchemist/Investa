import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Sequence, Tuple
import threading
import config
from db_utils import get_db_connection

# --- price basis -----------------------------------------------------------
#
# Yahoo's `Close` is split-adjusted as of the moment it was downloaded, so a
# stored series is only coherent with itself until the next split. `BASIS_RAW`
# marks a symbol whose rows hold the price actually quoted on the day, with the
# splits kept as events in `corporate_action` and applied at read time — the
# same convention the transaction ledger already uses for trades.
#
# The basis is a property of the SYMBOL, never of a row: a per-row flag would
# let one series hold both bases at once, which is exactly the seam this is
# meant to prevent.
BASIS_SPLIT_ADJ = "split_adj"  # legacy: as Yahoo served it, adjusted as-of-fetch
BASIS_RAW = "raw"  # archive: price as quoted, adjust on read

# Read-time adjustments.
ADJUST_NONE = "none"  # price as quoted on the day
ADJUST_SPLIT = "split"  # comparable across splits (the default, and today's behaviour)
ADJUST_TOTAL_RETURN = "total_return"  # splits + dividends reinvested
VALID_ADJUSTMENTS = (ADJUST_NONE, ADJUST_SPLIT, ADJUST_TOTAL_RETURN)

# Sanity bounds on a split ratio. Deliberately very wide.
#
# These were 0.01-1000, on the theory that a stray tiny value would otherwise
# "rescale an entire history". That reasoning was backwards and the bound did
# real damage: the ratio the provider reports is the authority for the
# adjustment it already applied to the prices it served, so recording it is what
# *explains* a rescale, and discarding it is what leaves one unexplained.
#
# The Tier C backfill rejected 55 ratios, every one a reverse split steeper than
# 100:1 — ordinary for serial-diluting micro-caps. ABVC was the proof: Yahoo
# reported 0.0002 on 2015-08-13 and its stored close falls from 1,719.75 to
# 0.3439 on that exact date, a real 5,000:1 reverse split it had not
# back-adjusted. Dropping the action turned one explained event into 122
# unexplained discontinuities, with 2005 prices reading over a million.
#
# So: mirror what the provider says, and let the verifier catch a genuine
# absurdity. Only a zero, a negative or a value beyond any conceivable corporate
# action is refused.
_MIN_SPLIT_RATIO, _MAX_SPLIT_RATIO = 1e-6, 1e6


class MarketDatabase:
    """
    Manages a persistent SQLite database for historical market data.
    Provides methods for upserting, querying, and checking data integrity.
    """

    # Class-level lock to serialize WRITES across threadpool workers.
    # When market_data.db lives on a cloud-synced path (e.g. Google Drive),
    # db_utils forces journal_mode=DELETE, which requires exclusive locking
    # for writes. Multiple concurrent threadpool tasks trying to upsert the
    # same file would otherwise hit "database is locked".
    # Reads (get_ohlcv, get_ohlcv_batch, get_fx) don't take this lock —
    # SQLite handles concurrent reads natively.
    _write_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(
                config.get_app_data_dir(), config.DB_DIR, "market_data.db"
            )
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        """Returns a thread-local database connection using the centralized helper."""
        return get_db_connection(self.db_path)

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._write_lock, self._get_connection() as conn:
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
                    source TEXT DEFAULT 'yahoo',
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
                    source TEXT DEFAULT 'yahoo',
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

    def upsert_ohlcv(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "1d",
        source: str = "yahoo",
        force: bool = False,
    ):
        """
        Upserts OHLCV data from a DataFrame.
        DataFrame must have a DatetimeIndex.

        `source` records who served the bar. It matters because bars no longer
        come from one place: a repair adjudicated against an independent
        reference rewrites them, and without provenance a corrected bar and an
        original one are indistinguishable.

        `force` overwrites a bar whatever its provenance. Reserved for a
        deliberate revert; a routine sync must never need it.
        """
        if df.empty:
            return

        n = len(df)

        # --- Vectorized column extraction (avoids slow per-row df.iterrows +
        # Series.get, which dominated cold /history time). Column-level
        # fallbacks and NaN->None semantics match the previous row loop. ---

        # Dates: format the index once. DatetimeIndex.strftime is vectorized;
        # fall back to the old per-element logic for non-datetime indexes.
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            date_strs = idx.strftime("%Y-%m-%d").tolist()
        else:
            date_strs = [
                ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                for ts in idx
            ]

        def clean_floats(col):
            """Coerce a column to a list of native floats with NaN -> None.
            `col` is a column name (str) or None when the column is absent."""
            if col is None or col not in df.columns:
                return [None] * n
            vals = (
                pd.to_numeric(df[col], errors="coerce")
                .to_numpy(dtype="float64")
                .tolist()
            )
            return [None if v != v else v for v in vals]  # v != v -> NaN

        opens = clean_floats("Open")
        highs = clean_floats("High")
        lows = clean_floats("Low")
        # close: prefer 'Close', else 'price' column (column-level fallback).
        closes = clean_floats("Close" if "Close" in df.columns else "price")
        # adj_close: prefer 'Adj Close', else NULL.
        # It used to fall back to a copy of `close`, which made the column a
        # lie: 27% of stored rows had adj_close == close, not because no
        # adjustment applied but because none was available. No consumer trusts
        # it (both get_historical_data and valuation_history deliberately read
        # raw Close), and total return is now derived from corporate_action, so
        # an honest NULL beats a fabricated number.
        adj = clean_floats("Adj Close") if "Adj Close" in df.columns else [None] * n

        if "Volume" in df.columns:
            vnum = (
                pd.to_numeric(df["Volume"], errors="coerce")
                .to_numpy(dtype="float64")
                .tolist()
            )
            vols = [0 if v != v else int(v) for v in vnum]
        else:
            vols = [0] * n

        params = [
            (
                symbol,
                date_strs[i],
                opens[i],
                highs[i],
                lows[i],
                closes[i],
                adj[i],
                vols[i],
                interval,
                source,
            )
            for i in range(n)
        ]

        with self._write_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # A database that predates the provenance migration still takes
                # bars; it just cannot say where they came from.
                if self._has_column(conn, "daily_ohlcv", "source"):
                    # A bar repaired against an independent reference must
                    # survive the next routine fetch from the provider that got
                    # it wrong. Yahoo keeps serving BYND's pre-split basis, so
                    # `INSERT OR REPLACE` quietly undid the correction the next
                    # time anything opened the chart — 1,836 bars, back to a
                    # series that steps 30x in the middle.
                    #
                    # So an ordinary write updates a bar of its own source (or
                    # one that predates provenance), and steps over an
                    # adjudicated one. `force=True` is how a deliberate revert
                    # still wins.
                    if force:
                        cursor.executemany(
                            """
                            INSERT OR REPLACE INTO daily_ohlcv
                            (symbol, date, open, high, low, close, adj_close,
                             volume, interval, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            params,
                        )
                    else:
                        cursor.executemany(
                            """
                            INSERT INTO daily_ohlcv
                            (symbol, date, open, high, low, close, adj_close,
                             volume, interval, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(symbol, date, interval) DO UPDATE SET
                                open = excluded.open, high = excluded.high,
                                low = excluded.low, close = excluded.close,
                                adj_close = excluded.adj_close,
                                volume = excluded.volume,
                                source = excluded.source
                            WHERE daily_ohlcv.source IS NULL
                               OR daily_ohlcv.source = excluded.source
                               OR daily_ohlcv.source = 'yahoo'
                        """,
                            params,
                        )
                else:
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO daily_ohlcv
                        (symbol, date, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        [row[:9] for row in params],
                    )
            except Exception as e_ins:
                logging.error(f"DB Upsert Error for {symbol} ({n} rows): {e_ins}")

            conn.commit()

            # Update sync metadata.
            # NB: INSERT OR REPLACE would blank price_basis/delisted_at on every
            # sync, silently reverting a converted symbol to the legacy basis.
            # Upsert only the column this method owns.
            now = datetime.now().isoformat()
            cursor.execute(
                """
                INSERT INTO sync_metadata (symbol, last_synced) VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET last_synced = excluded.last_synced
            """,
                (symbol, now),
            )
            conn.commit()

        # Corporate actions ride along on the same fetch (the worker requests
        # actions=True), so persist them rather than dropping the columns.
        try:
            self.upsert_actions(symbol, df)
        except Exception as exc:
            logging.warning(f"Actions upsert failed for {symbol}: {exc}")

    # --- corporate actions -------------------------------------------------

    def upsert_actions(
        self, symbol: str, df: pd.DataFrame, source: str = "yahoo"
    ) -> int:
        """
        Persist the `Dividends` / `Stock Splits` columns of a fetched frame.

        The worker already downloads with `actions=True`, so these columns ride
        along on every history fetch and were simply being dropped. Storing them
        is what makes read-time adjustment possible, and it costs no extra
        network call.

        Returns the number of event rows written.
        """
        if df is None or df.empty:
            return 0

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            date_strs = idx.strftime("%Y-%m-%d").tolist()
        else:
            date_strs = [
                ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                for ts in idx
            ]

        now = datetime.now().isoformat()
        params: List[tuple] = []

        if "Stock Splits" in df.columns:
            ratios = pd.to_numeric(df["Stock Splits"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for i, ratio in enumerate(ratios):
                # 0 and NaN both mean "no split on this day".
                if not ratio or ratio != ratio or ratio == 1.0:
                    continue
                if not (_MIN_SPLIT_RATIO <= ratio <= _MAX_SPLIT_RATIO):
                    logging.warning(
                        f"Actions: ignoring out-of-range split ratio {ratio} for "
                        f"{symbol} on {date_strs[i]}"
                    )
                    continue
                params.append(
                    (symbol, date_strs[i], "split", float(ratio), None, source, now)
                )

        if "Dividends" in df.columns:
            divs = pd.to_numeric(df["Dividends"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for i, amount in enumerate(divs):
                if not amount or amount != amount or amount < 0:
                    continue
                params.append(
                    (symbol, date_strs[i], "dividend", float(amount), None, source, now)
                )

        if not params:
            return 0

        with self._write_lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO corporate_action
                (symbol, date, kind, value, currency, source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            conn.commit()
        return len(params)

    def get_actions(
        self, symbols: Sequence[str], kind: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """{symbol: DataFrame[date, kind, value]} for the given symbols."""
        symbols = list(symbols)
        if not symbols:
            return {}

        placeholders = ", ".join(["?"] * len(symbols))
        query = (
            f"SELECT symbol, date, kind, value FROM corporate_action "
            f"WHERE symbol IN ({placeholders})"
        )
        params: List = list(symbols)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY symbol, date ASC"

        with self._get_connection() as conn:
            frame = pd.read_sql_query(query, conn, params=params)

        if frame.empty:
            return {}
        return {sym: group for sym, group in frame.groupby("symbol")}

    # --- price basis -------------------------------------------------------

    def get_price_basis(self, symbols: Sequence[str]) -> Dict[str, str]:
        """
        {symbol: basis}. Symbols absent from sync_metadata report the legacy
        basis, which is what an unconverted series actually holds.
        """
        symbols = list(symbols)
        if not symbols:
            return {}

        placeholders = ", ".join(["?"] * len(symbols))
        out = {s: BASIS_SPLIT_ADJ for s in symbols}
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"SELECT symbol, price_basis FROM sync_metadata "
                    f"WHERE symbol IN ({placeholders})",
                    symbols,
                )
                for symbol, basis in cursor:
                    if basis:
                        out[symbol] = basis
        except Exception as exc:
            # Pre-migration database: the column does not exist yet. Every
            # symbol is legacy, which is the default already in `out`.
            logging.debug(f"price_basis unavailable ({exc}); assuming legacy basis")
        return out

    def set_price_basis(self, symbol: str, basis: str, conn=None) -> None:
        """Set a symbol's basis. Pass `conn` to join a caller's transaction."""
        if basis not in (BASIS_RAW, BASIS_SPLIT_ADJ):
            raise ValueError(f"Unknown price basis: {basis}")

        sql = """
            INSERT INTO sync_metadata (symbol, price_basis) VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET price_basis = excluded.price_basis
        """
        if conn is not None:
            conn.execute(sql, (symbol, basis))
            return
        with self._write_lock, self._get_connection() as own:
            own.execute(sql, (symbol, basis))
            own.commit()

    # --- read-time adjustment ---------------------------------------------

    @staticmethod
    def _future_split_factors(
        date_strs: Sequence[str], splits: Optional[pd.DataFrame]
    ) -> Optional[np.ndarray]:
        """
        For each date d, the product of every split ratio with an ex-date
        strictly after d.

        A 4:1 split means the price quoted before it is four times the
        post-split basis, so:

            split_adjusted(d) = raw(d) / factor(d)
            raw(d)            = split_adjusted(d) * factor(d)

        Returns None when there is nothing to apply, so callers can skip the
        arithmetic entirely — the common case, since most symbols never split.
        """
        if splits is None or splits.empty or not len(date_strs):
            return None

        rows = splits[splits["kind"] == "split"] if "kind" in splits.columns else splits
        if rows.empty:
            return None

        ratios = pd.to_numeric(rows["value"], errors="coerce").to_numpy(dtype="float64")
        dates = rows["date"].astype(str).to_numpy()
        keep = np.isfinite(ratios) & (ratios > 0) & (ratios != 1.0)
        if not keep.any():
            return None
        ratios, dates = ratios[keep], dates[keep]

        order = np.argsort(dates, kind="stable")
        ratios, dates = ratios[order], dates[order]

        # suffix_products[i] = product of ratios[i:]; a date landing at index i
        # has exactly those splits still ahead of it.
        suffix = np.concatenate([np.cumprod(ratios[::-1])[::-1], [1.0]])

        # yyyy-MM-dd sorts lexicographically, so string search is correct here.
        # side='right' makes a split on date d itself count as already applied,
        # matching the ex-date convention.
        positions = np.searchsorted(
            dates, np.asarray(date_strs, dtype=object), side="right"
        )
        factors = suffix[positions]
        return factors if not np.allclose(factors, 1.0) else None

    @staticmethod
    def _dividend_factors(
        date_strs: Sequence[str],
        split_adjusted_close: np.ndarray,
        dividends: Optional[pd.DataFrame],
    ) -> Optional[np.ndarray]:
        """
        Back-adjustment factors that reinvest dividends, CRSP style.

        A dividend D going ex on date t drops the price by D, so prices before t
        are scaled by (1 - D / P_prev) to make the series continuous in total
        return terms. `split_adjusted_close` must already be on one split basis,
        and the dividend is scaled onto that basis alongside it.
        """
        if dividends is None or dividends.empty or not len(date_strs):
            return None

        rows = (
            dividends[dividends["kind"] == "dividend"]
            if "kind" in dividends.columns
            else dividends
        )
        if rows.empty:
            return None

        days = np.asarray(date_strs, dtype=object)
        factors = np.ones(len(days), dtype="float64")
        applied = False

        for _, row in rows.iterrows():
            ex_date = str(row["date"])
            amount = float(row["value"]) if row["value"] is not None else 0.0
            if amount <= 0:
                continue

            position = int(np.searchsorted(days, ex_date, side="left"))
            if position <= 0 or position >= len(days):
                # Ex-date outside the window; nothing before it to adjust.
                continue

            prev_close = split_adjusted_close[position - 1]
            if not np.isfinite(prev_close) or prev_close <= 0:
                continue

            ratio = 1.0 - (amount / prev_close)
            if not (0.0 < ratio <= 1.0):
                # A dividend at or above the prior close is bad data, not a
                # distribution; skipping beats zeroing the history before it.
                logging.debug(
                    f"Dividend factor skipped: {amount} vs prev close {prev_close} on {ex_date}"
                )
                continue

            factors[:position] *= ratio
            applied = True

        return factors if applied else None

    def _adjust_frame(
        self,
        frame: pd.DataFrame,
        symbol: str,
        stored_basis: str,
        adjust: str,
        actions: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Move `frame` from `stored_basis` onto the basis `adjust` asks for.

        The default (`ADJUST_SPLIT` over a legacy `split_adj` symbol) is a
        deliberate no-op: that is precisely what is stored today, so every
        existing caller keeps its current numbers until the symbol is converted.
        """
        if frame.empty:
            return frame

        price_cols = [
            c
            for c in ("Open", "High", "Low", "Close", "Adj Close")
            if c in frame.columns
        ]
        if not price_cols:
            return frame

        # Both bases express splits; only `none` wants them undone. So the net
        # split factor is 1 whenever the stored basis and the requested one
        # agree about splits — computed as a single factor rather than a
        # multiply followed by a divide, which would round-trip through float
        # twice for no reason.
        stored_is_split_adj = stored_basis == BASIS_SPLIT_ADJ
        wants_split_adj = adjust in (ADJUST_SPLIT, ADJUST_TOTAL_RETURN)

        needs_split_work = stored_is_split_adj != wants_split_adj
        needs_dividends = adjust == ADJUST_TOTAL_RETURN

        if not needs_split_work and not needs_dividends:
            return frame

        date_strs = frame.index.strftime("%Y-%m-%d").tolist()
        factors = (
            self._future_split_factors(date_strs, actions) if needs_split_work else None
        )

        if factors is not None:
            # stored split_adj -> raw multiplies; raw -> split_adj divides.
            net = factors if stored_is_split_adj else 1.0 / factors
            for col in price_cols:
                frame[col] = frame[col].to_numpy(dtype="float64") * net
            if "Volume" in frame.columns:
                frame["Volume"] = frame["Volume"].to_numpy(dtype="float64") / net

        if needs_dividends and "Close" in frame.columns:
            div_factors = self._dividend_factors(
                date_strs, frame["Close"].to_numpy(dtype="float64"), actions
            )
            if div_factors is not None:
                for col in price_cols:
                    frame[col] = frame[col].to_numpy(dtype="float64") * div_factors

        return frame

    def upsert_fx(
        self,
        pair: str,
        df: pd.DataFrame,
        interval: str = "1d",
        source: str = "yahoo",
    ):
        """
        Upserts FX rate data from a DataFrame.
        DataFrame must have a DatetimeIndex and a column 'Close' or 'rate'.
        """
        if df.empty:
            return

        col = (
            "Close"
            if "Close" in df.columns
            else (df.columns[0] if not df.empty else None)
        )
        if not col:
            return

        rows: List[Tuple[str, float]] = []
        for timestamp, row in df.iterrows():
            date_str = (
                timestamp.strftime("%Y-%m-%d")
                if hasattr(timestamp, "strftime")
                else str(timestamp)[:10]
            )
            rows.append((date_str, row[col]))
        self.upsert_fx_rows(pair, rows, interval=interval, source=source)

    def upsert_fx_rows(
        self,
        pair: str,
        rows: Sequence[Tuple[str, float]],
        interval: str = "1d",
        source: str = "yahoo",
        fill_only: bool = False,
    ) -> int:
        """
        Store (yyyy-MM-dd, rate) pairs, returning how many rows were written.

        `fill_only` writes a day only if the archive has none, which is how a
        second provider is allowed to contribute. The ECB's 14:15 CET fix and
        Yahoo's close differ by ~0.2% on an ordinary day; letting one overwrite
        the other would move every historical portfolio figure by that much
        while making neither series more true. Gaps are the exception — there
        the alternative is not a slightly different rate, it is the last known
        rate carried forward for weeks.

        `float()` is not decoration. sqlite3 has no adapter for a numpy scalar
        but numpy scalars implement the buffer protocol, so one stores silently
        as an 8-byte BLOB and reads back as bytes — which is exactly what
        happened to `USD=X`, whose synthetic identity rate of 1 sat in the
        archive as b'\\x01\\x00...' rather than 1.0.
        """
        clean: List[Tuple[str, str, float, str, str]] = []
        for day, rate in rows:
            if rate is None:
                continue
            try:
                value = float(rate)
            except (TypeError, ValueError):
                continue
            if value != value or value <= 0:  # NaN or nonsense
                continue
            clean.append((pair, str(day)[:10], value, interval, source))
        if not clean:
            return 0

        verb = "INSERT OR IGNORE" if fill_only else "INSERT OR REPLACE"
        with self._write_lock, self._get_connection() as conn:
            # A database that predates the provenance migration still takes
            # rates; it just cannot say where they came from.
            if self._has_column(conn, "daily_fx", "source"):
                sql = f"""{verb} INTO daily_fx (pair, date, rate, interval, source)
                          VALUES (?, ?, ?, ?, ?)"""
                params = clean
            else:
                sql = f"""{verb} INTO daily_fx (pair, date, rate, interval)
                          VALUES (?, ?, ?, ?)"""
                params = [row[:4] for row in clean]
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            written = cursor.rowcount
            conn.commit()
        return written if written is not None and written >= 0 else 0

    def _has_column(self, conn, table: str, column: str) -> bool:
        """Cached PRAGMA lookup — schema shape does not change under a process."""
        key = (table, column)
        cache = self.__dict__.setdefault("_column_cache", {})
        if key not in cache:
            cache[key] = any(
                row[1] == column for row in conn.execute(f"PRAGMA table_info({table})")
            )
        return cache[key]

    def get_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
        adjust: str = ADJUST_SPLIT,
    ) -> pd.DataFrame:
        """
        Retrieves OHLCV data for a symbol within a date range.

        `adjust` selects the basis the caller wants — see VALID_ADJUSTMENTS. The
        default reproduces exactly what this method has always returned.
        """
        if adjust not in VALID_ADJUSTMENTS:
            raise ValueError(
                f"Unknown adjustment {adjust!r}; expected one of {VALID_ADJUSTMENTS}"
            )

        query = """
            SELECT date, open, high, low, close, adj_close, volume
            FROM daily_ohlcv
            WHERE symbol = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=(symbol, interval, str(start_date), str(end_date))
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            # Rename columns to standard YF format for compatibility
            df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            # FORCE NUMERIC TYPES
            # This is critical because if 'Open' contains None (which sqlite returns for NULL),
            # pandas treats the column as 'object', causing interpolation to fail/warn.
            cols_to_numeric = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            basis = self.get_price_basis([symbol]).get(symbol, BASIS_SPLIT_ADJ)
            if basis != BASIS_SPLIT_ADJ or adjust != ADJUST_SPLIT:
                actions = self.get_actions([symbol]).get(symbol)
                df = self._adjust_frame(df, symbol, basis, adjust, actions)

        return df

    def get_ohlcv_batch(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
        adjust: str = ADJUST_SPLIT,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieves OHLCV data for multiple symbols in a collection of DataFrames.

        `adjust` behaves as in get_ohlcv. Actions and bases are read once for
        the whole batch, and symbols needing no adjustment skip the arithmetic.
        """
        if adjust not in VALID_ADJUSTMENTS:
            raise ValueError(
                f"Unknown adjustment {adjust!r}; expected one of {VALID_ADJUSTMENTS}"
            )
        if not symbols:
            return {}

        placeholders = ", ".join(["?"] * len(symbols))
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
            df_all["date"] = pd.to_datetime(df_all["date"])

            bases = self.get_price_basis(symbols)
            # Only symbols that actually need work pay for an actions lookup:
            # a legacy symbol read at the default adjustment is a pure no-op,
            # which is the overwhelmingly common case.
            needs_work = [
                s
                for s in df_all["symbol"].unique()
                if bases.get(s, BASIS_SPLIT_ADJ) != BASIS_SPLIT_ADJ
                or adjust != ADJUST_SPLIT
            ]
            actions_by_symbol = self.get_actions(needs_work) if needs_work else {}

            # Group by symbol and process each
            for sym, group in df_all.groupby("symbol"):
                df = group.drop(columns=["symbol"])
                df.set_index("date", inplace=True)
                df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

                # FORCE NUMERIC
                cols_to_numeric = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                ]
                for c in cols_to_numeric:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                if sym in needs_work:
                    df = self._adjust_frame(
                        df,
                        sym,
                        bases.get(sym, BASIS_SPLIT_ADJ),
                        adjust,
                        actions_by_symbol.get(sym),
                    )

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

        with self._write_lock, self._get_connection() as conn:
            cursor = conn.cursor()
            for timestamp, row in df.iterrows():
                # Store full timestamp ISO string
                ts_str = timestamp.isoformat()

                # Helper to clean NaNs
                def clean(val):
                    if pd.isna(val):
                        return None
                    return float(val)

                # Normalize columns
                open_val = clean(row.get("Open"))
                high_val = clean(row.get("High"))
                low_val = clean(row.get("Low"))
                close_val = clean(row.get("Close", row.get("price")))
                # Intraday usually doesn't have Adj Close diffs, but store if present
                adj_close_val = clean(row.get("Adj Close", close_val))
                volume_val = (
                    int(row.get("Volume", 0)) if pd.notna(row.get("Volume", 0)) else 0
                )

                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO intraday_ohlcv 
                        (symbol, timestamp, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            ts_str,
                            open_val,
                            high_val,
                            low_val,
                            close_val,
                            adj_close_val,
                            volume_val,
                            interval,
                        ),
                    )
                except Exception as e_ins:
                    logging.error(
                        f"DB Intraday Upsert Error for {symbol} on {ts_str}: {e_ins}"
                    )

            conn.commit()

    def get_intraday(
        self, symbol: str, start_ts: datetime, end_ts: datetime, interval: str
    ) -> pd.DataFrame:
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
            df = pd.read_sql_query(
                query, conn, params=(symbol, interval, start_str, end_str)
            )

        if not df.empty:
            # Fix for parsing mixed format ISO strings with timezone info
            # "YYYY-MM-DDTHH:MM:SS+00:00" might fail with default parser in some pandas versions
            # Enforce utc=True to avoid mixed naive/aware comparisons and FutureWarnings
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], format="ISO8601", errors="coerce", utc=True
            )
            df.set_index("timestamp", inplace=True)
            df.index.name = "Date"  # Standardize name for portfolio_logic compatibility

            # Rename columns to standard YF format
            df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

            # FORCE NUMERIC TYPES
            cols_to_numeric = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # --- data quality ------------------------------------------------------

    def get_data_quality(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Dict[str, Dict]:
        """Per-symbol summary of known defects in the stored price history.

        Populated by `scripts/flag_data_quality.py` from the same two checks
        that have always existed and only ever printed to a terminal — the
        split-consistency check and the archive verifier. The point of surfacing
        it is that a user looking at a chart cannot otherwise tell that the line
        steps 30x in the middle for no reason.

        Returns {} when the table has never been built. That is a normal state
        for a fresh clone, not an error: the table is derived and rebuildable,
        so nothing downstream should require it to exist.
        """
        query = "SELECT symbol, kind, severity, occurred_on, detail FROM data_quality"
        params: List = []
        # `None` means "everything flagged"; an empty sequence means "these zero
        # symbols", and the two must not collapse into each other — a client
        # asking about an empty holdings list would otherwise light up every row.
        if symbols is not None:
            wanted = [s for s in symbols if s]
            if not wanted:
                return {}
            query += f" WHERE symbol IN ({','.join('?' * len(wanted))})"
            params = list(wanted)

        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
        except Exception as exc:  # table absent on an archive never scanned
            logging.debug(f"data_quality unavailable ({exc})")
            return {}

        out: Dict[str, Dict] = {}
        for symbol, kind, severity, occurred_on, detail in rows:
            entry = out.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "severity": "medium",
                    "findings": 0,
                    "kinds": [],
                    "occurred_on": None,
                    "detail": None,
                },
            )
            entry["findings"] += 1
            if kind not in entry["kinds"]:
                entry["kinds"].append(kind)
            # 'high' means a split is on record that the prices do not reflect:
            # definitely wrong, as opposed to unexplained. It wins, and it is
            # the finding whose detail is worth showing.
            if severity == "high" and entry["severity"] != "high":
                entry["severity"] = "high"
                entry["detail"] = detail
                entry["occurred_on"] = occurred_on
            elif entry["detail"] is None:
                entry["detail"] = detail
                entry["occurred_on"] = occurred_on
        return out

    def get_fx(
        self, pair: str, start_date: date, end_date: date, interval: str = "1d"
    ) -> pd.DataFrame:
        """Retrieves FX rate data for a pair within a date range."""
        query = """
            SELECT date, rate 
            FROM daily_fx 
            WHERE pair = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=(pair, interval, str(start_date), str(end_date))
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.columns = [
                "price"
            ]  # Map to 'price' which is expected by portfolio_logic
        return df

    # --- fund NAVs ---------------------------------------------------------

    def upsert_fund_nav(
        self,
        fund_code: str,
        rows: List[Tuple[str, float]],
        currency: str = "THB",
        source: str = "sec_th",
    ) -> int:
        """
        Store (nav_date, nav_per_unit) pairs for a fund.

        Thai SSF/RMF NAVs have no commercial provider, so these come from the
        SEC's own daily-info API. Without them a held fund is valued from a
        single hand-entered number for its whole history.
        """
        params = [
            (fund_code, day, float(nav), currency, source)
            for day, nav in rows
            if day and nav is not None
        ]
        if not params:
            return 0

        with self._write_lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO fund_nav
                (fund_code, date, nav, currency, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                params,
            )
            conn.commit()
        return len(params)

    def get_fund_nav(
        self, fund_code: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """NAV series for a fund, shaped like get_fx (a 'price' column)."""
        query = """
            SELECT date, nav
            FROM fund_nav
            WHERE fund_code = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=(fund_code, str(start_date), str(end_date))
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.columns = ["price"]
        return df

    def get_latest_fund_navs(self) -> Dict[str, Tuple[str, float]]:
        """{FUND_CODE: (date, nav)} for the newest NAV on record per fund.

        Keys are upper-cased because callers match against ledger symbols, which
        the engine normalizes, while the codes here came from the override file
        and keep whatever case the user typed.

        This is the *current* price for a fund with no market feed. Without it
        the summary falls back to the hand-entered override, which is a scalar
        somebody typed once — SCBRCTECH sat 15% above its real NAV that way,
        while the graph beside it was already drawing the published series.
        """
        query = """
            SELECT f.fund_code, f.date, f.nav
            FROM fund_nav AS f
            JOIN (
                SELECT fund_code, MAX(date) AS date
                FROM fund_nav GROUP BY fund_code
            ) AS newest
              ON f.fund_code = newest.fund_code AND f.date = newest.date
        """
        out: Dict[str, Tuple[str, float]] = {}
        try:
            with self._get_connection() as conn:
                for code, day, nav in conn.execute(query):
                    if nav is None:
                        continue
                    out[str(code).upper().strip()] = (str(day), float(nav))
        except Exception as exc:
            logging.debug(f"fund_nav unavailable ({exc})")
        return out

    def get_fund_nav_coverage(self) -> Dict[str, Tuple[str, str, int]]:
        """{fund_code: (first_date, last_date, row_count)} — for backfill status."""
        query = """
            SELECT fund_code, MIN(date), MAX(date), COUNT(*)
            FROM fund_nav GROUP BY fund_code
        """
        out: Dict[str, Tuple[str, str, int]] = {}
        try:
            with self._get_connection() as conn:
                for code, first, last, count in conn.execute(query):
                    out[code] = (first, last, count)
        except Exception as exc:
            logging.debug(f"fund_nav unavailable ({exc})")
        return out

    # --- share counts ------------------------------------------------------

    def upsert_share_counts(
        self, rows: Dict[str, float], as_of: date, source: str = "yahoo"
    ) -> int:
        """Store {symbol: shares outstanding} observed on `as_of`."""
        params = [
            (symbol, float(shares), as_of.isoformat(), source)
            for symbol, shares in rows.items()
            if shares and float(shares) > 0
        ]
        if not params:
            return 0
        with self._write_lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO share_count (symbol, shares, as_of, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    shares = excluded.shares,
                    as_of = excluded.as_of,
                    source = excluded.source
                """,
                params,
            )
            conn.commit()
        return len(params)

    def get_share_counts(
        self, symbols: Sequence[str], max_age_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        {symbol: shares} for entries no older than `max_age_days`.

        Shares outstanding moves on buybacks and issuance — quarterly events —
        so a stale-by-a-week figure is fine and re-downloading the universe
        daily for it is not.
        """
        symbols = list(symbols)
        if not symbols:
            return {}

        query = f"SELECT symbol, shares, as_of FROM share_count WHERE symbol IN ({', '.join(['?'] * len(symbols))})"
        cutoff = None
        if max_age_days is not None:
            cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()

        out: Dict[str, float] = {}
        try:
            with self._get_connection() as conn:
                for symbol, shares, as_of in conn.execute(query, symbols):
                    if cutoff and (as_of or "") < cutoff:
                        continue
                    if shares:
                        out[symbol] = float(shares)
        except Exception as exc:
            logging.debug(f"share_count unavailable ({exc})")
        return out

    def get_latest_closes(
        self, symbols: Sequence[str], as_of: Optional[date] = None
    ) -> Dict[str, float]:
        """
        {symbol: most recent close at or before `as_of`}.

        One grouped query rather than a per-symbol read: the ranking asks for
        the whole universe at once.
        """
        symbols = list(symbols)
        if not symbols:
            return {}

        placeholders = ", ".join(["?"] * len(symbols))
        params: List = list(symbols)
        bound = ""
        if as_of:
            bound = "AND date <= ?"
            params.append(as_of.isoformat())

        query = f"""
            SELECT symbol, close FROM daily_ohlcv
            WHERE (symbol, date) IN (
                SELECT symbol, MAX(date) FROM daily_ohlcv
                WHERE symbol IN ({placeholders}) AND interval = '1d'
                  AND close IS NOT NULL {bound}
                GROUP BY symbol
            ) AND interval = '1d'
        """
        out: Dict[str, float] = {}
        with self._get_connection() as conn:
            for symbol, close in conn.execute(query, params):
                if close:
                    out[symbol] = float(close)
        return out

    def get_last_date(self, symbol: str, table: str = "daily_ohlcv") -> Optional[date]:
        """Returns the most recent date available in the DB for a symbol."""
        col = "symbol" if table == "daily_ohlcv" else "pair"
        query = f"SELECT MAX(date) FROM {table} WHERE {col} = ?"
        with self._get_connection() as conn:
            res = conn.execute(query, (symbol,)).fetchone()
            if res and res[0]:
                return datetime.strptime(res[0], "%Y-%m-%d").date()
        return None

    def get_last_dates(
        self, symbols: List[str], table: str = "daily_ohlcv"
    ) -> Dict[str, date]:
        """Returns a dict of {symbol: last_date} for a list of symbols."""
        col = "symbol" if table == "daily_ohlcv" else "pair"
        placeholders = ", ".join(["?"] * len(symbols))
        query = f"SELECT {col}, MAX(date) FROM {table} WHERE {col} IN ({placeholders}) GROUP BY {col}"

        results = {}
        with self._get_connection() as conn:
            cursor = conn.execute(query, symbols)
            for row in cursor:
                if row[1]:
                    results[row[0]] = datetime.strptime(row[1], "%Y-%m-%d").date()
        return results

    def get_first_dates(
        self, symbols: List[str], table: str = "daily_ohlcv"
    ) -> Dict[str, date]:
        """Returns a dict of {symbol: first_date} for a list of symbols."""
        col = "symbol" if table == "daily_ohlcv" else "pair"
        placeholders = ", ".join(["?"] * len(symbols))
        query = f"SELECT {col}, MIN(date) FROM {table} WHERE {col} IN ({placeholders}) GROUP BY {col}"

        results = {}
        with self._get_connection() as conn:
            cursor = conn.execute(query, symbols)
            for row in cursor:
                if row[1]:
                    results[row[0]] = datetime.strptime(row[1], "%Y-%m-%d").date()
        return results

    def get_sync_metadata_batch(self, symbols: List[str]) -> Dict[str, datetime]:
        """Returns a dict of {symbol: last_synced_datetime} for a list of symbols."""
        placeholders = ", ".join(["?"] * len(symbols))
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

    def check_integrity(
        self, symbol: str, new_df: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        """
        Compares new_df with existing DB data for overlapping dates.
        Returns (is_consistent, reason).
        If inconsistencies are found (e.g., adj_close differed significantly),
        it suggests a re-fetch of history.
        """
        if new_df.empty:
            return True, None

        # Never compare against a bar that is still moving. While a market is
        # open the stored row for today is rewritten every few minutes, so an
        # ordinary intraday tick reads as an "inconsistency" and triggers a full
        # multi-decade re-download. Measured on a single history call: ASML,
        # GOOG, GOOGL and NOW all fired at 0.13-0.37% drift, purely from the
        # clock. Only settled sessions can be checked.
        try:
            from utils_time import get_est_today

            live_date = get_est_today().isoformat()
        except Exception:
            live_date = date.today().isoformat()

        overlap_dates = [
            d for d in new_df.index.strftime("%Y-%m-%d").tolist() if d < live_date
        ]
        if not overlap_dates:
            return True, None

        placeholders = ", ".join(["?"] * len(overlap_dates))
        # Adjudicated bars are excluded from the comparison, and must be. They
        # were corrected *because* they disagree with what this provider serves,
        # so comparing them against it reports the correction as corruption —
        # and the response is a full multi-decade re-download that used to
        # overwrite the repair. Opening BYND's one-year chart did exactly that:
        # a year of Yahoo bars 30x away from the repaired ones, read as a
        # broken archive, and 1,836 corrected bars refetched away.
        with self._get_connection() as conn:
            source_guard = (
                " AND (source IS NULL OR source = 'yahoo')"
                if self._has_column(conn, "daily_ohlcv", "source")
                else ""
            )
            query = f"""
                SELECT date, adj_close, close
                FROM daily_ohlcv
                WHERE symbol = ? AND date IN ({placeholders}){source_guard}
            """
            db_data = pd.read_sql_query(query, conn, params=[symbol] + overlap_dates)

        if db_data.empty:
            return True, None  # No overlap, no conflict

        # Vectorized comparison. The check now runs for every symbol rather than
        # only held ones, so the old per-row .loc[] lookup would have been paid
        # on the whole archive nightly.
        incoming = new_df.copy()
        incoming["_day"] = incoming.index.strftime("%Y-%m-%d")
        new_col = "Adj Close" if "Adj Close" in incoming.columns else "Close"
        if new_col not in incoming.columns:
            return True, None

        merged = db_data.merge(
            incoming[["_day", new_col]].rename(columns={new_col: "_new"}),
            left_on="date",
            right_on="_day",
            how="inner",
        )
        if merged.empty:
            return True, None

        # Compare like with like: the stored adj_close is a copy of close for
        # 27% of rows, so fall back to close when it is absent.
        db_vals = pd.to_numeric(merged["adj_close"], errors="coerce")
        db_vals = db_vals.where(
            db_vals.notna(), pd.to_numeric(merged["close"], errors="coerce")
        )
        new_vals = pd.to_numeric(merged["_new"], errors="coerce")

        valid = db_vals.notna() & new_vals.notna() & (db_vals != 0)
        if not valid.any():
            return True, None

        diffs = ((db_vals - new_vals).abs() / db_vals.abs()).where(valid)
        worst = diffs.max()
        if pd.notna(worst) and worst > 0.001:
            row = merged.loc[diffs.idxmax()]
            return (
                False,
                f"Inconsistency detected on {row['date']}: DB={db_vals[diffs.idxmax()]}, "
                f"New={new_vals[diffs.idxmax()]} (diff={worst:.4%})",
            )

        return True, None
