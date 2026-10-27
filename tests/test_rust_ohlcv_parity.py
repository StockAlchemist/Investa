"""Phase 8: the daily price reads (investa_core.read_daily_ohlcv behind
market_db.get_ohlcv_batch / get_ohlcv) against the originals frozen in
tests/reference/ohlcv_read.py.

Bitwise and type for type: the dict's symbols and their order, each frame's
columns, dtypes, index (dtype, name, freq) and every cell, exceptions included.
The random archives are written cell by cell, so they hold what the reader
must either reproduce or decline:
- NULL prices and volumes, a column entirely NULL, prices stored as integers;
- fractional volumes;
- text in a price;
- dates that are impossible, in another shape or outside datetime64[ns];
- symbols in mixed case, with punctuation, or not ASCII;
- another interval;
- price bases and corporate actions, so the adjustments run on top.
Requests cover:
- symbols that are missing, duplicated, or passed in a tuple or as non-strings;
- date bounds given as dates, datetimes or strings, or reversed;
- every adjustment.
"""

import math
import os
import sqlite3
import struct
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import market_db  # noqa: E402
from reference import ohlcv_read as ref  # noqa: E402

SYMBOLS = ["AAPL", "aapl", "BRK-B", "^GSPC", "THB=X", "PTT.BK", "Ünï", "MSFT", "Z"]


def _same_value(a, b, path):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _same_frame(a, b, path):
    assert type(a) is type(b), path
    assert list(a.columns) == list(b.columns), f"{path}: columns"
    assert a.columns.dtype == b.columns.dtype, f"{path}: column index dtype"
    assert list(a.dtypes) == list(b.dtypes), (
        f"{path}: {list(a.dtypes)} vs {list(b.dtypes)}"
    )
    assert type(a.index) is type(b.index), f"{path}: index type"
    assert a.index.equals(b.index) and a.index.dtype == b.index.dtype, f"{path}: index"
    assert a.index.name == b.index.name, f"{path}: index name"
    assert getattr(a.index, "freq", None) == getattr(b.index, "freq", None), path
    for col in a.columns:
        for i, (x, y) in enumerate(zip(a[col].tolist(), b[col].tolist())):
            _same_value(x, y, f"{path}[{col}][{i}]")


def _same(a, b, path):
    if isinstance(b, dict):
        assert type(a) is dict and list(a) == list(b), f"{path}: {list(a)} vs {list(b)}"
        for k in b:
            _same_frame(a[k], b[k], f"{path}[{k}]")
    else:
        _same_frame(a, b, path)


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:
        return ("raised", type(exc))


def _with_archive_schema(db):
    """The Phase 1 archive tables (scripts/migrate_market_archive_phase1.py),
    as tests/test_market_archive.py applies them."""
    with db._get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS corporate_action (
                symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL,
                value REAL NOT NULL, currency TEXT, source TEXT NOT NULL,
                ingested_at TEXT NOT NULL, PRIMARY KEY (symbol, date, kind));
            """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sync_metadata)")}
        if "price_basis" not in cols:
            conn.execute(
                "ALTER TABLE sync_metadata ADD COLUMN price_basis TEXT "
                "NOT NULL DEFAULT 'split_adj'"
            )
        conn.commit()
    return db


def _archive(tmp_path, rng, name, clean=False):
    db = _with_archive_schema(
        market_db.MarketDatabase(db_path=str(tmp_path / f"{name}.db"))
    )
    rows = []
    odd = 0.99 if clean else rng.random()  # which defects this archive carries
    for sym in rng.choice(SYMBOLS, int(rng.integers(1, len(SYMBOLS))), replace=False):
        days = pd.bdate_range("2019-01-01", periods=int(rng.integers(1, 300)))
        for d in days:
            day = d.strftime("%Y-%m-%d")
            r = rng.random()
            if odd < 0.08 and r < 0.01:
                day = str(
                    rng.choice(
                        [
                            "2019-02-30",
                            "2019-1-05",
                            "2019-01-05 00:00:00",
                            "1600-01-03",
                            "x",
                        ]
                    )
                )
            cells = [float(rng.lognormal(4, 0.3)) for _ in range(5)]
            vol = int(rng.integers(0, 10**7))
            if odd < 0.4 and rng.random() < 0.05:
                cells[int(rng.integers(0, 5))] = None
            if odd < 0.5 and rng.random() < 0.05 and cells[0] is not None:
                cells[0] = float(int(cells[0]))  # REAL affinity: an integral price
            if 0.4 < odd < 0.5 and rng.random() < 0.05:
                vol = None
            if 0.5 < odd < 0.6 and rng.random() < 0.05:
                vol = float(vol) + 0.5
            if 0.6 < odd < 0.66 and rng.random() < 0.01:
                cells[2] = "n/a"  # stays TEXT under REAL affinity
            if 0.66 < odd < 0.72:
                cells[1] = None  # a column entirely NULL
            interval = "1wk" if rng.random() < 0.03 else "1d"
            rows.append((str(sym), day, *cells, vol, interval))
    con = sqlite3.connect(db.db_path)
    con.executemany(
        "INSERT OR REPLACE INTO daily_ohlcv (symbol, date, open, high, low, close, adj_close, volume, interval)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    present = sorted({r[0] for r in rows})
    for sym in present:
        if rng.random() < 0.3:
            db.set_price_basis(sym, market_db.BASIS_RAW)
        if rng.random() < 0.4:
            idx = pd.DatetimeIndex(
                pd.bdate_range("2019-01-01", periods=300)[rng.integers(0, 300, 3)]
            )
            db.upsert_actions(
                sym,
                pd.DataFrame(
                    {"Dividends": [0.5, 0.0, 0.3], "Stock Splits": [0.0, 2.0, 0.0]},
                    index=idx,
                ),
            )
    return db


def _requests(rng):
    for _ in range(14):
        k = int(rng.integers(0, 5))
        syms = [str(s) for s in rng.choice(SYMBOLS + ["NONE"], k)]
        r = rng.random()
        if r < 0.05:
            syms = tuple(syms)
        elif r < 0.08:
            syms = syms + [None]
        start = (
            date(2019, 1, 1)
            + pd.Timedelta(days=int(rng.integers(-30, 400))).to_pytimedelta()
        )
        end = start + pd.Timedelta(days=int(rng.integers(-5, 500))).to_pytimedelta()
        r = rng.random()
        if r < 0.1:
            start, end = datetime(start.year, start.month, start.day, 9), str(end)
        interval = "1wk" if rng.random() < 0.05 else "1d"
        adjust = str(
            rng.choice(
                [market_db.ADJUST_SPLIT] * 3
                + [market_db.ADJUST_NONE, market_db.ADJUST_TOTAL_RETURN]
            )
        )
        yield syms, start, end, interval, adjust


@pytest.mark.parametrize("seed", range(8))
def test_reads_match(tmp_path, seed):
    assert market_db._rust_reader_bound(), (
        "the Rust reader did not bind to Python's SQLite"
    )
    rng = np.random.default_rng(7500 + seed)
    for trial in range(6):
        db = _archive(tmp_path, rng, f"a{seed}_{trial}")
        for syms, start, end, interval, adjust in _requests(rng):
            path = f"{seed}/{trial} {syms!r} {start}..{end} {interval} {adjust}"
            want = _outcome(ref.get_ohlcv_batch, db, syms, start, end, interval, adjust)
            got = _outcome(db.get_ohlcv_batch, syms, start, end, interval, adjust)
            assert got[0] == want[0], (path, want, got)
            if want[0] == "ok":
                _same(got[1], want[1], path)
            else:
                assert got[1] is want[1], path
            for sym in list(syms)[:2]:
                want = _outcome(ref.get_ohlcv, db, sym, start, end, interval, adjust)
                got = _outcome(db.get_ohlcv, sym, start, end, interval, adjust)
                assert got[0] == want[0], (path, sym, want, got)
                if want[0] == "ok":
                    _same(got[1], want[1], f"{path} single {sym}")
                else:
                    assert got[1] is want[1], path


def test_the_reader_declines_what_it_cannot_read(tmp_path):
    db = market_db.MarketDatabase(db_path=str(tmp_path / "odd.db"))
    con = sqlite3.connect(db.db_path)
    con.executemany(
        "INSERT INTO daily_ohlcv (symbol, date, open, high, low, close, adj_close, volume, interval)"
        " VALUES (?,?,?,?,?,?,?,?,'1d')",
        [("A", "2020-01-02", 1.0, 1.0, 1.0, 1.0, 1.0, 5), ("A", "2020-01-03", "n/a", 1.0, 1.0, 1.0, 1.0, 5),
         ("B", "2020-01-02 00:00:00", 1.0, 1.0, 1.0, 1.0, 1.0, 5)],
    )  # fmt: skip
    con.commit()
    con.close()
    for syms in (["A"], ["B"]):
        assert (
            market_db._read_daily_frames(
                db.db_path, syms, "1d", "2020-01-01", "2020-12-31"
            )
            is None
        )
    assert (
        market_db._read_daily_frames(
            db.db_path, ("A",), "1d", "2020-01-01", "2020-12-31"
        )
        is None
    )
    assert (
        market_db._read_daily_frames(
            str(tmp_path / "missing" / "x.db"), ["A"], "1d", "2020-01-01", "2020-12-31"
        )
        is None
    )


def test_frames_do_not_share_memory(tmp_path):
    rng = np.random.default_rng(7600)
    db = _archive(tmp_path, rng, "share", clean=True)
    assert market_db._read_daily_frames(
        db.db_path, list(SYMBOLS), "1d", "2019-01-01", "2020-06-01"
    )
    frames = db.get_ohlcv_batch(list(SYMBOLS), date(2019, 1, 1), date(2020, 6, 1))
    before = {k: v.copy() for k, v in frames.items()}
    first = next(iter(frames.values()))
    first.iloc[:, :] = 0
    for k in list(frames)[1:]:
        _same_frame(frames[k], before[k], k)
