"""Tests for the raw-price archive: actions capture and read-time adjustment.

The conversion's whole safety claim is that reading a symbol at the default
adjustment reproduces exactly what was stored before, so the compatibility test
here is as important as the correctness ones.

These run against a temporary database, not the user's archive.
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from market_db import (  # noqa: E402
    ADJUST_NONE,
    ADJUST_SPLIT,
    ADJUST_TOTAL_RETURN,
    BASIS_RAW,
    BASIS_SPLIT_ADJ,
    MarketDatabase,
)


@pytest.fixture
def db(tmp_path):
    """A MarketDatabase on a throwaway file, with the Phase 1 schema applied."""
    database = MarketDatabase(str(tmp_path / "market_test.db"))
    with database._get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS corporate_action (
                symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL,
                value REAL NOT NULL, currency TEXT, source TEXT NOT NULL,
                ingested_at TEXT NOT NULL, PRIMARY KEY (symbol, date, kind));
            CREATE TABLE IF NOT EXISTS share_count (
                symbol TEXT NOT NULL, shares REAL NOT NULL, as_of TEXT NOT NULL,
                source TEXT NOT NULL, PRIMARY KEY (symbol));
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sync_metadata)")}
        if "price_basis" not in cols:
            conn.execute(
                "ALTER TABLE sync_metadata ADD COLUMN price_basis TEXT "
                "NOT NULL DEFAULT 'split_adj'"
            )
        conn.commit()
    return database


def _frame(dates, closes, dividends=None, splits=None):
    index = pd.to_datetime(dates)
    data = {
        "Open": closes,
        "High": closes,
        "Low": closes,
        "Close": closes,
        "Adj Close": closes,
        "Volume": [1000] * len(closes),
    }
    if dividends is not None:
        data["Dividends"] = dividends
    if splits is not None:
        data["Stock Splits"] = splits
    return pd.DataFrame(data, index=index)


# --- actions capture -------------------------------------------------------


def test_splits_and_dividends_are_captured_from_the_price_frame(db):
    """The columns the worker already fetches must land in corporate_action."""
    frame = _frame(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 100.0, 25.0],
        dividends=[0.0, 0.5, 0.0],
        splits=[0.0, 0.0, 4.0],
    )
    written = db.upsert_actions("TEST", frame)
    assert written == 2

    actions = db.get_actions(["TEST"])["TEST"]
    splits = actions[actions["kind"] == "split"]
    dividends = actions[actions["kind"] == "dividend"]

    assert list(splits["date"]) == ["2024-01-04"]
    assert list(splits["value"]) == [4.0]
    assert list(dividends["date"]) == ["2024-01-03"]
    assert list(dividends["value"]) == [0.5]


def test_extreme_but_real_reverse_splits_are_kept(db):
    """
    A 5,000:1 reverse split is ordinary for a serial-diluting micro-cap, and the
    provider's ratio is the authority for the adjustment it already applied. An
    earlier 0.01 floor discarded 55 of these in one backfill; ABVC's real
    2015-08-13 event (close 1,719.75 -> 0.3439) then read as 122 unexplained
    discontinuities because nothing was left to explain it.
    """
    frame = _frame(["2015-08-13"], [0.3439], splits=[0.0002])
    assert db.upsert_actions("ABVC", frame) == 1
    stored = db.get_actions(["ABVC"])["ABVC"]
    assert stored.iloc[0]["value"] == pytest.approx(0.0002)


def test_ratios_beyond_any_corporate_action_are_still_rejected(db):
    """The bound is a sanity check, not a judgement about plausible splits."""
    for absurd in (0.0, -4.0, 1e9):
        frame = _frame(["2024-01-02"], [100.0], splits=[absurd])
        assert db.upsert_actions("TEST", frame) == 0
    assert db.get_actions(["TEST"]) == {}


def test_upsert_actions_is_idempotent(db):
    frame = _frame(["2024-01-04"], [25.0], splits=[4.0])
    db.upsert_actions("TEST", frame)
    db.upsert_actions("TEST", frame)
    assert len(db.get_actions(["TEST"])["TEST"]) == 1


# --- split factor arithmetic ----------------------------------------------


def test_future_split_factors_compound_and_respect_the_ex_date():
    splits = pd.DataFrame(
        {
            "date": ["2020-08-31", "2024-01-05"],
            "kind": ["split", "split"],
            "value": [4.0, 2.0],
        }
    )
    factors = MarketDatabase._future_split_factors(
        ["2020-08-27", "2020-08-31", "2024-01-04", "2024-01-05"], splits
    )
    # Before both splits: 4*2. On the first ex-date it is already applied, so
    # only the later split remains ahead. On the second ex-date, nothing.
    assert list(factors) == [8.0, 2.0, 2.0, 1.0]


def test_no_splits_means_no_adjustment_work():
    assert MarketDatabase._future_split_factors(["2024-01-02"], None) is None
    empty = pd.DataFrame({"date": [], "kind": [], "value": []})
    assert MarketDatabase._future_split_factors(["2024-01-02"], empty) is None


# --- the compatibility guarantee ------------------------------------------


def test_legacy_symbol_at_default_adjustment_returns_stored_values(db):
    """
    The default read over an unconverted symbol must be a pure no-op — this is
    what lets the migration ship without moving a single existing number.
    """
    closes = [10.0, 11.0, 12.0, 3.25]
    frame = _frame(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        closes,
        splits=[0.0, 0.0, 0.0, 4.0],
    )
    db.upsert_ohlcv("TEST", frame)
    db.upsert_actions("TEST", frame)

    got = db.get_ohlcv("TEST", date(2024, 1, 1), date(2024, 1, 31))
    assert list(got["Close"]) == closes


def test_split_round_trip_is_lossless(db):
    """raw -> split-adjusted -> raw must return the original quoted prices."""
    quoted = [400.0, 404.0, 100.0]  # 4:1 split on the third day
    frame = _frame(
        ["2024-01-03", "2024-01-04", "2024-01-05"], quoted, splits=[0.0, 0.0, 4.0]
    )
    db.upsert_ohlcv("TEST", frame)
    db.upsert_actions("TEST", frame)
    db.set_price_basis("TEST", BASIS_RAW)

    window = (date(2024, 1, 1), date(2024, 1, 31))
    adjusted = db.get_ohlcv("TEST", *window, adjust=ADJUST_SPLIT)
    assert list(adjusted["Close"]) == [100.0, 101.0, 100.0]

    back = db.get_ohlcv("TEST", *window, adjust=ADJUST_NONE)
    assert list(back["Close"]) == quoted


def test_a_split_added_later_does_not_rewrite_stored_rows(db):
    """
    The point of the whole exercise: recording a new split must change what
    reads return without touching a single stored bar.
    """
    quoted = [400.0, 404.0]
    db.upsert_ohlcv("TEST", _frame(["2024-01-03", "2024-01-04"], quoted))
    db.set_price_basis("TEST", BASIS_RAW)

    window = (date(2024, 1, 1), date(2024, 1, 31))
    assert list(db.get_ohlcv("TEST", *window, adjust=ADJUST_SPLIT)["Close"]) == quoted

    # The split happens tomorrow and is recorded as an event.
    db.upsert_actions("TEST", _frame(["2024-01-05"], [100.0], splits=[4.0]))

    assert list(db.get_ohlcv("TEST", *window, adjust=ADJUST_SPLIT)["Close"]) == [
        100.0,
        101.0,
    ]
    # ... while the archive itself is untouched.
    assert list(db.get_ohlcv("TEST", *window, adjust=ADJUST_NONE)["Close"]) == quoted


def test_volume_moves_opposite_to_price(db):
    frame = _frame(["2024-01-03"], [400.0])
    db.upsert_ohlcv("TEST", frame)
    db.upsert_actions("TEST", _frame(["2024-01-05"], [100.0], splits=[4.0]))
    db.set_price_basis("TEST", BASIS_RAW)

    got = db.get_ohlcv("TEST", date(2024, 1, 1), date(2024, 1, 31), adjust=ADJUST_SPLIT)
    assert got["Volume"].iloc[0] == pytest.approx(4000.0)


# --- total return ----------------------------------------------------------


def test_total_return_reinvests_dividends(db):
    """A dividend scales earlier prices by (1 - D/prev_close)."""
    frame = _frame(["2024-01-03", "2024-01-04"], [100.0, 99.0], dividends=[0.0, 1.0])
    db.upsert_ohlcv("TEST", frame)
    db.upsert_actions("TEST", frame)
    db.set_price_basis("TEST", BASIS_RAW)

    window = (date(2024, 1, 1), date(2024, 1, 31))
    plain = db.get_ohlcv("TEST", *window, adjust=ADJUST_SPLIT)
    total = db.get_ohlcv("TEST", *window, adjust=ADJUST_TOTAL_RETURN)

    assert list(plain["Close"]) == [100.0, 99.0]
    # 1.00 on a 100.00 prior close -> earlier prices scale by 0.99.
    assert total["Close"].iloc[0] == pytest.approx(99.0)
    assert total["Close"].iloc[1] == pytest.approx(99.0)


def test_unknown_adjustment_is_rejected(db):
    with pytest.raises(ValueError):
        db.get_ohlcv("TEST", date(2024, 1, 1), date(2024, 1, 31), adjust="nonsense")


# --- basis bookkeeping -----------------------------------------------------


def test_sync_does_not_reset_a_converted_symbols_basis(db):
    """
    upsert_ohlcv used INSERT OR REPLACE on sync_metadata, which drops every
    column it does not name — silently reverting a converted symbol to the
    legacy basis on its next sync.
    """
    db.upsert_ohlcv("TEST", _frame(["2024-01-03"], [400.0]))
    db.set_price_basis("TEST", BASIS_RAW)

    db.upsert_ohlcv("TEST", _frame(["2024-01-04"], [404.0]))

    assert db.get_price_basis(["TEST"])["TEST"] == BASIS_RAW


def test_unknown_symbols_report_the_legacy_basis(db):
    assert db.get_price_basis(["NEVER_SEEN"])["NEVER_SEEN"] == BASIS_SPLIT_ADJ


# --- integrity check -------------------------------------------------------


def test_integrity_check_ignores_the_still_moving_session(db, monkeypatch):
    """
    Today's bar is rewritten every few minutes while a market is open, so
    comparing against it reported ordinary intraday drift as corruption and
    triggered a full multi-decade refetch.
    """
    import utils_time

    # check_integrity asks the market clock, not the wall clock, so that is
    # what has to be pinned.
    monkeypatch.setattr(utils_time, "get_est_today", lambda: date(2024, 1, 5))

    db.upsert_ohlcv("TEST", _frame(["2024-01-04", "2024-01-05"], [100.0, 100.0]))

    # A wildly different price for *today* must not trip the check...
    live = _frame(["2024-01-05"], [130.0])
    consistent, _ = db.check_integrity("TEST", live)
    assert consistent

    # ... while the same divergence on a settled session must.
    settled = _frame(["2024-01-04"], [130.0])
    consistent, reason = db.check_integrity("TEST", settled)
    assert not consistent
    assert "2024-01-04" in reason


# --- share counts and latest closes ----------------------------------------


def test_latest_close_is_the_most_recent_bar(db):
    db.upsert_ohlcv("AAA", _frame(["2024-01-02", "2024-01-05"], [10.0, 12.0]))
    db.upsert_ohlcv("BBB", _frame(["2024-01-03"], [20.0]))

    closes = db.get_latest_closes(["AAA", "BBB", "MISSING"])
    assert closes == {"AAA": 12.0, "BBB": 20.0}


def test_latest_close_respects_an_as_of_bound(db):
    db.upsert_ohlcv("AAA", _frame(["2024-01-02", "2024-01-05"], [10.0, 12.0]))
    assert db.get_latest_closes(["AAA"], as_of=date(2024, 1, 3)) == {"AAA": 10.0}


def test_share_counts_round_trip(db):
    written = db.upsert_share_counts({"AAA": 1_000_000.0}, date(2024, 1, 2))
    assert written == 1
    assert db.get_share_counts(["AAA"]) == {"AAA": 1_000_000.0}


def test_share_counts_past_the_age_limit_are_not_served(db):
    """
    A stale figure must look absent so the caller re-fetches it. Shares
    outstanding moves quarterly, so the window is generous — but not infinite.
    """
    db.upsert_share_counts({"AAA": 1.0}, date(2020, 1, 1))
    assert db.get_share_counts(["AAA"], max_age_days=30) == {}
    assert db.get_share_counts(["AAA"]) == {"AAA": 1.0}  # no limit, still there


def test_share_counts_update_in_place(db):
    db.upsert_share_counts({"AAA": 1.0}, date(2024, 1, 2))
    db.upsert_share_counts({"AAA": 2.0}, date(2024, 6, 2))
    assert db.get_share_counts(["AAA"]) == {"AAA": 2.0}


def test_nonpositive_share_counts_are_ignored(db):
    assert db.upsert_share_counts({"AAA": 0.0, "BBB": -5.0}, date(2024, 1, 2)) == 0
    assert db.get_share_counts(["AAA", "BBB"]) == {}


# --- bar provenance --------------------------------------------------------
#
# `daily_ohlcv` gained a `source` column once bars stopped coming from one
# place: 34,120 of them were rewritten from an independent reference, and
# without provenance a corrected bar and an original are indistinguishable.


def test_a_stored_bar_records_who_served_it(db):
    frame = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [100]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    db.upsert_ohlcv("TEST", frame)
    db.upsert_ohlcv("REPAIRED", frame, source="tiingo")

    with db._get_connection() as conn:
        rows = dict(conn.execute("SELECT symbol, source FROM daily_ohlcv"))
    assert rows["TEST"] == "yahoo", "the default names the feed everything came from"
    assert rows["REPAIRED"] == "tiingo"


def test_bars_still_store_on_an_archive_without_the_provenance_column(tmp_path):
    """An archive that predates the migration takes bars; it just cannot say
    where they came from."""
    import sqlite3

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE daily_ohlcv (symbol TEXT, date TEXT, open REAL, high REAL,"
        " low REAL, close REAL, adj_close REAL, volume INTEGER,"
        " interval TEXT DEFAULT '1d', PRIMARY KEY (symbol, date, interval))"
    )
    conn.execute(
        "CREATE TABLE sync_metadata (symbol TEXT PRIMARY KEY, last_synced TEXT,"
        " inception_date TEXT, info_json TEXT)"
    )
    conn.commit()
    conn.close()

    database = MarketDatabase(path)
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    database.upsert_ohlcv("TEST", frame, source="tiingo")

    with database._get_connection() as check:
        assert check.execute("SELECT close FROM daily_ohlcv").fetchone()[0] == 1.0
