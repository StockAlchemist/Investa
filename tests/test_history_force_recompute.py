"""`force=True` on the history path must bypass every cache, not just the fast one.

This guards a defect that made the golden gate report success without looking at
anything. `force` was accepted by `_calculate_historical_performance_internal`
and then never used, so a "real recompute" was served from the on-disk
daily-results feather — whose key is built from the transactions, the dates and
the display settings, and carries no component of the market data the series is
computed from.

The gate compares two runs on the same day. With the feather in the way it
handed back the identical frame twice and reported "portfolio unchanged"
whatever had happened underneath: an exchange rate replaced by 999.0 produced a
bit-identical portfolio value. Every archive change through Phases 1-4 was
gated by that comparison.

So both halves matter and are asserted separately: the in-memory SWR entry must
be skipped (or a second forced call would be served from the first), and
`use_daily_results_cache=False` must reach `calculate_historical_performance`
(or the recompute reads the feather and market data still cannot move it).
"""

import asyncio
import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import server.portfolio_service as ps  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    """Stand in for the expensive calculation, recording how it was asked for."""
    recorded = []

    def fake_calculate(**kwargs):
        recorded.append(kwargs)
        return (pd.DataFrame({"value": [float(len(recorded))]}), {}, {}, "ok")

    monkeypatch.setattr(ps, "calculate_historical_performance", fake_calculate)
    ps._PORTFOLIO_HISTORY_CACHE.clear()
    return recorded


def call(force: bool):
    return asyncio.run(
        ps._get_historical_performance_cached(
            df=pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")]}),
            manual_overrides_dict={},
            user_symbol_map={},
            user_excluded_symbols=set(),
            account_currency_map={},
            original_csv_file_path="/tmp/does-not-matter.db",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            interval="1d",
            benchmark_symbols_yf=[],
            display_currency="THB",
            include_accounts=None,
            account_cash_mode_map={},
            db_mtime=1.0,
            force=force,
        )
    )


def test_an_unforced_second_call_is_served_from_the_cache(calls):
    call(force=False)
    call(force=False)
    assert len(calls) == 1
    assert calls[0]["use_daily_results_cache"] is True


def test_forcing_recomputes_every_time(calls):
    call(force=False)
    call(force=True)
    call(force=True)
    assert len(calls) == 3, "a forced run must not be served the SWR entry"


def test_forcing_also_disables_the_on_disk_daily_results_cache(calls):
    """The half that made a 999.0 exchange rate invisible."""
    call(force=True)
    assert calls[0]["use_daily_results_cache"] is False


def test_a_forced_run_does_not_poison_the_cache_for_everyone_else(calls):
    """Forcing is a diagnostic. It must not write its result back as the entry
    that ordinary requests then read."""
    call(force=True)
    call(force=False)
    assert len(calls) == 2, "the forced result must not have been cached"
