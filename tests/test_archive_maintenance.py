"""Tests for the staleness-driven maintenance planner.

This runs on a laptop that sleeps, so the planner cannot assume it fires daily.
What matters is that it does the right thing after an hour, after a fortnight,
and when run twice in a row — which is what these cover.
"""

import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import archive_maintenance as am  # noqa: E402


def _setup(monkeypatch, *, newest_bar, inc_age_h, core_age_h, today=date(2026, 8, 26)):
    monkeypatch.setattr(am, "newest_bar", lambda: newest_bar)
    monkeypatch.setattr(am, "last_trading_day", lambda: today - timedelta(days=1))
    ages = {"incremental": inc_age_h, "core": core_age_h}
    monkeypatch.setattr(am, "newest_snapshot_age", lambda mode: ages[mode])


def _labels(jobs):
    return [label for label, _ in jobs]


def test_nothing_is_due_when_everything_is_fresh(monkeypatch):
    _setup(monkeypatch, newest_bar="2026-08-25", inc_age_h=1, core_age_h=8)
    jobs, skipped = am.plan(force=False)
    assert jobs == []
    assert len(skipped) == 4


def test_stale_prices_pull_the_split_check_with_them(monkeypatch):
    """A new bar can carry a new split, so the check only earns its keep after
    prices move — and must not be skipped when they do."""
    _setup(monkeypatch, newest_bar="2026-08-20", inc_age_h=1, core_age_h=8)
    jobs, _ = am.plan(force=False)
    labels = " ".join(_labels(jobs))
    assert "price delta" in labels
    assert "split-consistency" in labels


def test_the_delta_window_widens_to_cover_a_long_absence(monkeypatch):
    """Away a fortnight, ask for a fortnight — not the usual five days, which
    would leave a hole the next run has no reason to notice."""
    _setup(monkeypatch, newest_bar="2026-08-05", inc_age_h=1, core_age_h=8)
    jobs, _ = am.plan(force=False)
    delta = next(argv for label, argv in jobs if "price delta" in label)
    days = int(delta[delta.index("--days") + 1])
    assert days >= 20 + am.DELTA_OVERLAP_DAYS - 5


def test_the_window_is_capped_however_long_the_gap(monkeypatch):
    """A year away should not ask for a year of every symbol; that is a
    backfill, and it has its own command."""
    _setup(monkeypatch, newest_bar="2020-01-02", inc_age_h=1, core_age_h=8)
    jobs, _ = am.plan(force=False)
    delta = next(argv for label, argv in jobs if "price delta" in label)
    days = int(delta[delta.index("--days") + 1])
    assert days == am.DELTA_MAX_DAYS


def test_the_window_always_carries_an_overlap(monkeypatch):
    """Even one day behind, the request overlaps — that overlap is what
    check_integrity compares against."""
    _setup(monkeypatch, newest_bar="2026-08-24", inc_age_h=1, core_age_h=8)
    jobs, _ = am.plan(force=False)
    delta = next(argv for label, argv in jobs if "price delta" in label)
    days = int(delta[delta.index("--days") + 1])
    assert days >= am.DELTA_OVERLAP_DAYS


def test_an_old_incremental_snapshot_is_due_on_its_own(monkeypatch):
    """Snapshots do not depend on prices having moved: the small tables can
    change without a new bar."""
    _setup(monkeypatch, newest_bar="2026-08-25", inc_age_h=48, core_age_h=8)
    jobs, _ = am.plan(force=False)
    labels = _labels(jobs)
    assert any("incremental" in label for label in labels)
    assert not any("price delta" in label for label in labels)


def test_a_missing_snapshot_counts_as_due(monkeypatch):
    _setup(monkeypatch, newest_bar="2026-08-25", inc_age_h=None, core_age_h=None)
    labels = " ".join(_labels(am.plan(force=False)[0]))
    assert "incremental" in labels
    assert "core" in labels


def test_an_empty_archive_still_plans_a_fetch(monkeypatch):
    _setup(monkeypatch, newest_bar=None, inc_age_h=1, core_age_h=8)
    labels = " ".join(_labels(am.plan(force=False)[0]))
    assert "price delta" in labels


def test_force_runs_everything(monkeypatch):
    _setup(monkeypatch, newest_bar="2026-08-25", inc_age_h=1, core_age_h=8)
    labels = " ".join(_labels(am.plan(force=True)[0]))
    for expected in ("price delta", "split-consistency", "incremental", "core"):
        assert expected in labels


def test_last_trading_day_skips_the_weekend(monkeypatch):
    """Run on a Sunday, the newest bar anyone could have is Friday's; treating
    Saturday as the target would report the archive stale every weekend."""
    import archive_maintenance

    monkeypatch.setattr(
        archive_maintenance, "date", type("D", (), {"today": staticmethod(lambda: date(2026, 8, 30)),
                                                    "fromisoformat": staticmethod(date.fromisoformat)})
    )
    # Sunday 30 Aug 2026 -> Friday 28 Aug
    assert am.last_trading_day().weekday() < 5
