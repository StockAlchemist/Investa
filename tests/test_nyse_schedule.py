"""utils_time.nyse_schedule against the calendar it caches.

Every answer must be the frame `get_nyse_calendar().schedule()` gives for the
same arguments: same rows, columns, dtypes and index, and the same exception
for a reversed range. Covers ranges inside the cached window (empty, one day,
holidays, years long), across and outside its edges, and arguments that are
not plain dates.
"""

import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import utils_time  # noqa: E402


def _direct(start, end):
    try:
        return (
            "ok",
            utils_time.get_nyse_calendar().schedule(start_date=start, end_date=end),
        )
    except Exception as exc:
        return ("raised", type(exc))


def _cached(start, end):
    try:
        return ("ok", utils_time.nyse_schedule(start, end))
    except Exception as exc:
        return ("raised", type(exc))


def _same(a, b, path):
    assert a[0] == b[0], (path, a, b)
    if a[0] != "ok":
        assert a[1] is b[1], path
        return
    x, y = a[1], b[1]
    assert list(x.columns) == list(y.columns), path
    assert list(x.dtypes) == list(y.dtypes), path
    assert x.index.equals(y.index) and x.index.dtype == y.index.dtype, path
    assert x.index.name == y.index.name and x.index.freq == y.index.freq, path
    assert x.equals(y), path


def _ranges(rng):
    first, last = utils_time._full_schedule()[1:3]
    out = [
        (date(2025, 12, 25), date(2025, 12, 25)),  # a holiday
        (date(2026, 10, 24), date(2026, 10, 25)),  # a weekend
        (date(2025, 1, 10), date(2025, 1, 1)),  # reversed: raises
        (first, first + timedelta(days=9)),
        (last - timedelta(days=9), last),
        (first - timedelta(days=30), first + timedelta(days=30)),  # across the start
        (last - timedelta(days=30), last + timedelta(days=30)),  # across the end
        (date(1950, 1, 1), date(1950, 3, 1)),  # before the window
        (datetime(2025, 3, 3, 12), datetime(2025, 3, 20)),  # datetimes
        (pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-20")),
        ("2025-03-03", "2025-03-20"),
    ]
    span = (last - first).days
    for _ in range(120):
        start = first + timedelta(days=int(rng.integers(0, span)))
        length = int(rng.choice([0, 1, 3, 10, 40, 400, 4000]))
        out.append((start, min(last, start + timedelta(days=length))))
    return out


@pytest.mark.parametrize("seed", range(3))
def test_cached_schedule_matches_the_calendar(seed):
    rng = np.random.default_rng(7400 + seed)
    for start, end in _ranges(rng):
        _same(_cached(start, end), _direct(start, end), f"{start!r}..{end!r}")


def test_a_slice_is_a_copy():
    got = utils_time.nyse_schedule(date(2025, 3, 3), date(2025, 3, 7))
    got.iloc[0, 0] = pd.Timestamp("2000-01-01", tz="UTC")
    _same(
        _cached(date(2025, 3, 3), date(2025, 3, 7)),
        _direct(date(2025, 3, 3), date(2025, 3, 7)),
        "after mutating a slice",
    )
