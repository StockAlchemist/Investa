"""Phase 11a: db_utils' transactions and watchlists in Rust
(rust/investa_core/src/db_portfolio.rs) against the original frozen in
tests/reference/db_utils.py.

Twin databases take the same random sequence of calls, one through the
original and one through the port. After every call the two must agree on:
- the return value;
- every table's rows;
- every log record (level, message and the logged exception's type).
Timestamps the code stamps with `datetime.now()` are masked, since the two
sides run microseconds apart.

The calls cover:
- transactions added with each date shape (date, datetime, Timestamp, three
  string formats, an impossible day, garbage, None);
- Type casing and stray spaces, NaN and None cells, signed and unsigned
  totals for inflow and outflow types, repeated ExternalIDs;
- values SQLite cannot bind (a list);
- updates with unknown keys, invalid dates, missing ids, an empty update;
- deletes;
- every watchlist operation;
- loading, with blank, "<NA>" and missing currencies;
- the empty-database check.
"""

import datetime as dt
import logging
import math
import os
import re
import sqlite3
import struct
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import db_utils as new  # noqa: E402
from reference import db_utils as ref  # noqa: E402

STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?")


def _mask(x):
    if isinstance(x, str):
        return STAMP.sub("<now>", x)
    if isinstance(x, float):
        return ("f", "nan" if math.isnan(x) else struct.pack("<d", x))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_mask(v) for v in x])
    if isinstance(x, dict):
        return ("dict", [(k, _mask(v)) for k, v in x.items()])
    if isinstance(x, pd.DataFrame):
        return (
            "df",
            list(x.columns),
            list(map(str, x.dtypes)),
            [
                _mask(v)
                for v in x.astype(object)
                .where(x.notna(), None)
                .to_numpy()
                .ravel()
                .tolist()
            ],
        )
    return (type(x).__name__, x)


def _tables(conn):
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return [
        (n, _mask(conn.execute(f'SELECT * FROM "{n}" ORDER BY rowid').fetchall()))
        for n in names
    ]


def _call(fn, args, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        try:
            out = ("ok", fn(*args))
        except Exception as exc:
            out = ("raised", type(exc).__name__, str(exc))
    logs = [
        (
            r.levelname,
            STAMP.sub("<now>", r.getMessage()),
            r.exc_info[0].__name__ if r.exc_info else None,
        )
        for r in caplog.records
        if r.name == "root"
    ]
    return _mask(out), logs, [(w.category.__name__, str(w.message)) for w in warned]


def _value(rng):
    return [None, np.nan, 0, 1.5, -2.25, "x", "", 10, 125.5, float("nan")][
        int(rng.integers(0, 10))
    ]


def _date(rng):
    return [
        dt.date(2024, 3, 1), dt.datetime(2024, 3, 2, 15, 30), pd.Timestamp("2024-03-03"),
        "2024-03-04", "2024-03-05 10:00:00", "2024-03-06T11:00:00", "2024-02-30",
        "not a date", None, "2024-3-7",
    ][int(rng.integers(0, 10))]  # fmt: skip


def _transaction(rng):
    t = {
        "Date": _date(rng),
        "Type": str(
            rng.choice(
                [
                    "buy",
                    " Sell ",
                    "DIVIDEND",
                    "deposit",
                    "withdrawal",
                    "Fees",
                    "stock split",
                    "transfer",
                    "buy to cover",
                ]
            )
        ),
        "Symbol": str(rng.choice(["AAPL", "MSFT", "$CASH"])),
        "Quantity": _value(rng),
        "Price/Share": _value(rng),
        "Total Amount": [None, 100.0, -100.0, 0.0, "250", "abc", np.nan, 1e-12][
            int(rng.integers(0, 8))
        ],
        "Commission": _value(rng),
        "Account": str(rng.choice(["IBKR", "E*Trade"])),
        "Local Currency": str(rng.choice(["USD", "THB"])),
    }
    if rng.random() < 0.3:
        t["ExternalID"] = str(rng.choice(["ext-1", "ext-2", ""]))
    if rng.random() < 0.2:
        t["Note"] = str(rng.choice(["hello", "To: IBKR"]))
    if rng.random() < 0.05:
        t["Quantity"] = [1, 2]  # SQLite cannot bind a list
    if rng.random() < 0.1:
        t["Unknown"] = 5
    return t


def _pair(tmp_path, name):
    out = []
    for side in ("ref", "new"):
        conn = sqlite3.connect(str(tmp_path / f"{name}_{side}.db"))
        ref.create_transactions_table(conn)
        out.append(conn)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_match(tmp_path, seed, caplog):
    rng = np.random.default_rng(8100 + seed)
    a, b = _pair(tmp_path, f"s{seed}")
    for step in range(150):
        r = rng.random()
        if r < 0.3:
            t = _transaction(rng)
            commit = bool(rng.random() < 0.8)
            calls = [
                (ref.add_transaction_to_db, (a, dict(t))),
                (new.add_transaction_to_db, (b, dict(t))),
            ]
            calls = [
                (lambda f=f, c=c, d=d: f(c, d, commit=commit), ())
                for f, (c, d) in calls
            ]
        elif r < 0.42:
            upd = {k: v for k, v in _transaction(rng).items() if rng.random() < 0.4}
            tid = int(rng.integers(1, 12))
            calls = [
                (ref.update_transaction_in_db, (a, tid, dict(upd))),
                (new.update_transaction_in_db, (b, tid, dict(upd))),
            ]
        elif r < 0.48:
            tid = int(rng.integers(1, 12))
            calls = [
                (ref.delete_transaction_from_db, (a, tid)),
                (new.delete_transaction_from_db, (b, tid)),
            ]
        elif r < 0.56:
            name = str(rng.choice(["Tech", "Favorites", "favorites ", "Value"]))
            calls = [
                (ref.create_watchlist, (a, name)),
                (new.create_watchlist, (b, name)),
            ]
        elif r < 0.62:
            calls = [
                (ref.get_or_create_favorites_watchlist, (a,)),
                (new.get_or_create_favorites_watchlist, (b,)),
            ]
        elif r < 0.70:
            sym, wid = (
                str(rng.choice(["AAPL", "aapl ", "MSFT"])),
                int(rng.integers(1, 4)),
            )
            note = str(rng.choice(["", "watch"]))
            calls = [
                (ref.add_to_watchlist, (a, sym, note, wid)),
                (new.add_to_watchlist, (b, sym, note, wid)),
            ]
        elif r < 0.74:
            sym, wid = str(rng.choice(["AAPL", "MSFT"])), int(rng.integers(1, 4))
            calls = [
                (ref.remove_from_watchlist, (a, sym, wid)),
                (new.remove_from_watchlist, (b, sym, wid)),
            ]
        elif r < 0.78:
            wid, nm = int(rng.integers(1, 4)), str(rng.choice(["Renamed", "X"]))
            calls = [
                (ref.rename_watchlist, (a, wid, nm)),
                (new.rename_watchlist, (b, wid, nm)),
            ]
        elif r < 0.80:
            wid = int(rng.integers(1, 4))
            calls = [(ref.delete_watchlist, (a, wid)), (new.delete_watchlist, (b, wid))]
        elif r < 0.86:
            fn = str(
                rng.choice(
                    [
                        "get_all_watchlists",
                        "get_watchlist",
                        "get_watchlist_ids_for_symbol",
                    ]
                )
            )
            arg = {
                "get_all_watchlists": (),
                "get_watchlist": (int(rng.integers(1, 4)),),
                "get_watchlist_ids_for_symbol": (" aapl",),
            }[fn]
            calls = [(getattr(ref, fn), (a, *arg)), (getattr(new, fn), (b, *arg))]
        elif r < 0.94:
            cmap = {"IBKR": "usd"} if rng.random() < 0.5 else {}
            if rng.random() < 0.3:
                blank = str(rng.choice(["", "<NA>", " <na> "]))
                for c in (a, b):
                    c.execute(
                        'UPDATE transactions SET "Local Currency" = ? WHERE id % 3 = 0',
                        (blank,),
                    )
                    c.commit()
            calls = [
                (ref.load_all_transactions_from_db, (a, cmap, "THB")),
                (new.load_all_transactions_from_db, (b, cmap, "THB")),
            ]
        else:
            path = str(tmp_path / ("x.csv" if rng.random() < 0.5 else "missing.csv"))
            open(tmp_path / "x.csv", "w").close()
            calls = [
                (ref.check_if_db_empty_and_csv_exists, (a, path)),
                (new.check_if_db_empty_and_csv_exists, (b, path)),
            ]
        (fa, aa), (fb, ab) = calls
        got_a = _call(fa, aa, caplog)
        got_b = _call(fb, ab, caplog)
        assert got_a == got_b, (
            seed,
            step,
            fa.__name__ if hasattr(fa, "__name__") else fa,
            got_a,
            got_b,
        )
        assert _tables(a) == _tables(b), (seed, step)


def test_a_closed_connection_and_a_failing_statement(tmp_path, caplog):
    a, b = _pair(tmp_path, "closed")
    for c in (a, b):
        c.execute("DROP TABLE watchlists")
    for fn in (
        "get_all_watchlists",
        "create_watchlist",
        "rename_watchlist",
        "delete_watchlist",
    ):
        args = {
            "get_all_watchlists": (),
            "create_watchlist": ("X",),
            "rename_watchlist": (1, "Y"),
            "delete_watchlist": (1,),
        }[fn]
        assert _call(getattr(ref, fn), (a, *args), caplog) == _call(
            getattr(new, fn), (b, *args), caplog
        ), fn
    a.close(), b.close()
    closed_calls = [
        (
            "add_transaction_to_db",
            ({"Date": "2024-01-01", "Type": "buy", "Total Amount": 5.0},),
        ),
        ("delete_transaction_from_db", (3,)),
        ("get_watchlist", (1,)),
        ("load_all_transactions_from_db", ({}, "USD")),
    ]
    for fn, args in closed_calls:
        assert _call(getattr(ref, fn), (a, *args), caplog) == _call(
            getattr(new, fn), (b, *args), caplog
        ), fn
