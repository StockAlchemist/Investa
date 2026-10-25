import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, date
from typing import List, Dict, Optional, Sequence, Tuple
import threading
import config
import investa_core
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


def _iso_days(idx: pd.DatetimeIndex) -> list:
    """`idx.strftime("%Y-%m-%d").tolist()`, without the cost.

    `strftime` formats element by element through Python, and a history sync
    calls it four times per symbol on multi-decade indexes: 2.8s of a 19s cold
    /history, measured. Truncating to whole days in NumPy is ~15x faster.

    Same values, including the edge cases: a tz-aware index formats its local
    wall date (as strftime does), pre-1970 instants floor to their own day, and
    NaT stays NaN.
    """
    # Ported to Rust: rust/investa_core/src/mdb.rs
    return investa_core.mdb_iso_days(idx)


# --- the Rust reader for the daily archive ------------------------------------
#
# pd.read_sql_query builds a Python object per cell before pandas infers each
# column's dtype, and the date strings are parsed again after; a full history
# read is ~340k rows x 8 cells. investa_core steps the rows in C, through the
# same SQLite library Python's sqlite3 module uses (see
# rust/investa_core/src/sqlite_read.rs), and the frames are built from its
# column arrays. When it cannot read the rows exactly as pandas would, it says
# so and the pandas path below runs instead.

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_RUST_READER: Optional[bool] = None


def _rust_reader_bound() -> bool:
    # Ported to Rust: rust/investa_core/src/mdb.rs
    return investa_core.mdb_rust_reader_bound()


def _read_daily_frames(
    db_path: str, symbols, interval, start: str, end: str
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    {symbol: frame} for the daily rows, each frame as the pandas path builds
    it before any adjustment — date index, the six columns renamed, numeric —
    in the query's symbol order. None to take the pandas path: the reader is
    off, an argument is not what it binds verbatim, the database cannot be
    opened, or a value would come out differently.
    """
    # Ported to Rust: rust/investa_core/src/mdb.rs
    return investa_core.mdb_read_daily_frames(db_path, symbols, interval, start, end)


# The NumPy originals of the two factor helpers, for the inputs the Rust
# versions leave to them (a NaN day among the dates, an unsorted index, closes
# that are not a float64 array).


def _future_split_factors_numpy(
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


def _dividend_factors_numpy(
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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        investa_core.mdb_init_db(self)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        investa_core.mdb_upsert_ohlcv(self, symbol, df, interval, source, force)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_upsert_actions(self, symbol, df, source)

    def get_actions(
        self, symbols: Sequence[str], kind: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """{symbol: DataFrame[date, kind, value]} for the given symbols."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_actions(self, symbols, kind)

    # --- price basis -------------------------------------------------------

    def get_price_basis(self, symbols: Sequence[str]) -> Dict[str, str]:
        """
        {symbol: basis}. Symbols absent from sync_metadata report the legacy
        basis, which is what an unconverted series actually holds.
        """
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_price_basis(self, symbols)

    def set_price_basis(self, symbol: str, basis: str, conn=None) -> None:
        """Set a symbol's basis. Pass `conn` to join a caller's transaction."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        investa_core.mdb_set_price_basis(self, symbol, basis, conn)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_future_split_factors(date_strs, splits)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_dividend_factors(
            date_strs, split_adjusted_close, dividends
        )

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_adjust_frame(
            self, frame, symbol, stored_basis, adjust, actions
        )

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        investa_core.mdb_upsert_fx(self, pair, df, interval, source)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_upsert_fx_rows(
            self, pair, rows, interval, source, fill_only
        )

    def _has_column(self, conn, table: str, column: str) -> bool:
        """Cached PRAGMA lookup — schema shape does not change under a process."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_has_column(self, conn, table, column)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_ohlcv(
            self, symbol, start_date, end_date, interval, adjust
        )

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_ohlcv_batch(
            self, symbols, start_date, end_date, interval, adjust
        )

    def upsert_intraday(self, symbol: str, df: pd.DataFrame, interval: str):
        """
        Upserts Intraday OHLCV data from a DataFrame.
        DataFrame must have a DatetimeIndex.
        Dates are stored as ISO timestamps (YYYY-MM-DDTHH:MM:SS...).
        """
        # Ported to Rust: rust/investa_core/src/mdb.rs
        investa_core.mdb_upsert_intraday(self, symbol, df, interval)

    def get_intraday(
        self, symbol: str, start_ts: datetime, end_ts: datetime, interval: str
    ) -> pd.DataFrame:
        """Retrieves Intraday OHLCV data for a symbol within a timestamp range."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_intraday(self, symbol, start_ts, end_ts, interval)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_data_quality(self, symbols)

    def get_fx(
        self, pair: str, start_date: date, end_date: date, interval: str = "1d"
    ) -> pd.DataFrame:
        """Retrieves FX rate data for a pair within a date range."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_fx(self, pair, start_date, end_date, interval)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_upsert_fund_nav(self, fund_code, rows, currency, source)

    def get_fund_nav(
        self, fund_code: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """NAV series for a fund, shaped like get_fx (a 'price' column)."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_fund_nav(self, fund_code, start_date, end_date)

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
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_latest_fund_navs(self)

    def get_fund_nav_coverage(self) -> Dict[str, Tuple[str, str, int]]:
        """{fund_code: (first_date, last_date, row_count)} — for backfill status."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_fund_nav_coverage(self)

    # --- share counts ------------------------------------------------------

    def upsert_share_counts(
        self, rows: Dict[str, float], as_of: date, source: str = "yahoo"
    ) -> int:
        """Store {symbol: shares outstanding} observed on `as_of`."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_upsert_share_counts(self, rows, as_of, source)

    def get_share_counts(
        self, symbols: Sequence[str], max_age_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        {symbol: shares} for entries no older than `max_age_days`.

        Shares outstanding moves on buybacks and issuance — quarterly events —
        so a stale-by-a-week figure is fine and re-downloading the universe
        daily for it is not.
        """
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_share_counts(self, symbols, max_age_days)

    def get_latest_closes(
        self, symbols: Sequence[str], as_of: Optional[date] = None
    ) -> Dict[str, float]:
        """
        {symbol: most recent close at or before `as_of`}.

        One grouped query rather than a per-symbol read: the ranking asks for
        the whole universe at once.
        """
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_latest_closes(self, symbols, as_of)

    def get_last_date(self, symbol: str, table: str = "daily_ohlcv") -> Optional[date]:
        """Returns the most recent date available in the DB for a symbol."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_last_date(self, symbol, table)

    def get_last_dates(
        self, symbols: List[str], table: str = "daily_ohlcv"
    ) -> Dict[str, date]:
        """Returns a dict of {symbol: last_date} for a list of symbols."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_last_dates(self, symbols, table)

    def get_first_dates(
        self, symbols: List[str], table: str = "daily_ohlcv"
    ) -> Dict[str, date]:
        """Returns a dict of {symbol: first_date} for a list of symbols."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_first_dates(self, symbols, table)

    def get_sync_metadata_batch(self, symbols: List[str]) -> Dict[str, datetime]:
        """Returns a dict of {symbol: last_synced_datetime} for a list of symbols."""
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_get_sync_metadata_batch(self, symbols)

    def check_integrity(
        self, symbol: str, new_df: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        """
        Compares new_df with existing DB data for overlapping dates.
        Returns (is_consistent, reason).
        If inconsistencies are found (e.g., adj_close differed significantly),
        it suggests a re-fetch of history.
        """
        # Ported to Rust: rust/investa_core/src/mdb.rs
        return investa_core.mdb_check_integrity(self, symbol, new_df)
