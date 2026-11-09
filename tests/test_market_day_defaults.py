"""Defaults reckoned on the market's day, not the server's.

On the Bangkok host the server's date runs up to a day ahead of New York, so a
default that read `date.today()` could land in the next day, or the next
quarter, before the market got there.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config_manager  # noqa: E402
import display_config  # noqa: E402
import edgar_sic  # noqa: E402


def test_sec_quarters_follow_the_market_quarter(monkeypatch):
    # 31 March in New York is already 1 April in Bangkok: the quarter that has
    # just completed is still 2025q4, not 2026q1.
    monkeypatch.setattr(edgar_sic, "get_est_today", lambda: date(2026, 3, 31))
    assert edgar_sic.recent_quarters(3) == ["2025q4", "2025q3", "2025q2"]
    assert edgar_sic.recent_quarters(1, today=date(2026, 4, 1)) == ["2026q1"]


def test_default_graph_window_ends_on_the_market_day(monkeypatch):
    monkeypatch.setattr(display_config, "get_est_today", lambda: date(2026, 11, 6))
    start, end = display_config.default_graph_dates()
    assert end == date(2026, 11, 6)
    assert (end - start).days == display_config.DEFAULT_GRAPH_DAYS_AGO
    # Read afresh each time, not frozen when the module was imported.
    monkeypatch.setattr(display_config, "get_est_today", lambda: date(2026, 11, 9))
    assert display_config.default_graph_dates()[1] == date(2026, 11, 9)


def test_new_settings_default_to_that_window(monkeypatch, tmp_path):
    monkeypatch.setattr(display_config, "get_est_today", lambda: date(2026, 11, 6))
    defaults = config_manager.ConfigManager(str(tmp_path))._get_default_gui_config()
    assert defaults["graph_end_date"] == "2026-11-06"
    assert defaults["graph_start_date"] == "2024-11-06"
