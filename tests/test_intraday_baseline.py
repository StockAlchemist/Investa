"""The 1D graph anchors on the previous *session's* close.

Two rules keep the day change on the graph equal to the day change in the hero
panel: the anchor is picked from a day the market actually opened, and prices are
carried forward from the last real trade instead of being interpolated toward the
next one (which would price a past moment with a quote that hadn't happened yet).
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from portfolio_history import align_prices_to_grid  # noqa: E402
from server.portfolio_service import _previous_trading_day  # noqa: E402


def test_previous_trading_day_skips_the_weekend():
    # Monday 2026-08-10 -> Friday 2026-08-07, not Sunday.
    assert _previous_trading_day(date(2026, 8, 10)) == date(2026, 8, 7)


def test_previous_trading_day_skips_a_market_holiday():
    # Tuesday after MLK Day (Mon 2026-01-19) -> the preceding Friday.
    assert _previous_trading_day(date(2026, 1, 20)) == date(2026, 1, 16)


def test_previous_trading_day_of_a_weekday_is_the_day_before():
    assert _previous_trading_day(date(2026, 8, 7)) == date(2026, 8, 6)


def test_prices_carry_forward_across_a_closed_market():
    """A grid point between two sessions is worth the earlier close, never a
    blend with the later one."""
    friday_close = pd.Timestamp("2026-08-07 20:00", tz="UTC")
    sunday = pd.Timestamp("2026-08-09 20:00", tz="UTC")
    monday_open = pd.Timestamp("2026-08-10 13:30", tz="UTC")
    observed = {0: pd.Series([100.0, 110.0], index=[friday_close, monday_open])}

    aligned = align_prices_to_grid(
        observed, pd.DatetimeIndex([friday_close, sunday, monday_open])
    )

    # The Sunday point holds Friday's close — not 90-something on the way to 110.
    assert aligned[0].tolist() == [100.0, 100.0, 110.0]


def test_bar_between_grid_points_is_not_dropped():
    """A 15:45 close is the price at the 16:00 grid point, not a hole."""
    bar = pd.Timestamp("2026-08-10 19:45", tz="UTC")
    grid = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-10 19:30", tz="UTC"),
            pd.Timestamp("2026-08-10 20:00", tz="UTC"),
        ]
    )
    aligned = align_prices_to_grid({0: pd.Series([42.0], index=[bar])}, grid)

    assert not np.isnan(aligned[0]).any()
    assert aligned[0].iloc[-1] == 42.0
