"""Tests for the rule-based strategies and the trend signal.

The cases concentrate on the timing errors that would change the strategy
without breaking it. A signal that quietly reads a mid-month price, or that
lets the running month into its own moving average, still returns a plausible
"in" or "out" — it just implements a different (and worse) rule than the one
that was backtested. Those are the failures worth pinning down.
"""

import os
import re
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import strategies as st


def _daily_series(values_by_month, days_per_month=20, tz=None):
    """Build a daily close series where each month ends on the given value."""
    stamps, values = [], []
    for (year, month), close in values_by_month:
        for day in range(1, days_per_month + 1):
            stamps.append(pd.Timestamp(year=year, month=month, day=day, tz=tz))
            # Flat through the month, so the month-end close is unambiguous.
            values.append(close)
    return pd.Series(values, index=pd.DatetimeIndex(stamps))


def _rising(n_months, start=100.0, step=10.0, end_year=2026, end_month=6):
    """`n_months` of month-ends ending at (end_year, end_month), rising."""
    months = []
    year, month = end_year, end_month
    for i in range(n_months):
        months.append(((year, month), start + step * (n_months - 1 - i)))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return list(reversed(months))


# --- month-end extraction ---------------------------------------------------


def test_month_end_takes_the_last_actual_trading_day():
    """A month with no trading on the calendar last day keeps its real date."""
    stamps = pd.DatetimeIndex(["2026-01-28", "2026-01-29", "2026-02-02"])
    series = pd.Series([10.0, 11.0, 12.0], index=stamps)
    monthly = st.month_end_closes(series)
    assert list(monthly.values) == [11.0, 12.0]
    # January's row is the 29th — not an invented 31 January.
    assert str(monthly.index[0].date()) == "2026-01-29"


def test_month_end_handles_timezone_aware_index():
    """Exchange-stamped data arrives tz-aware; it must not raise or drop rows."""
    series = _daily_series(_rising(3), tz="America/New_York")
    monthly = st.month_end_closes(series)
    assert len(monthly) == 3


# --- the timing contract ----------------------------------------------------


def test_active_signal_ignores_the_running_month():
    """
    The active signal is set at the last *completed* month-end.

    A crash inside the current month must not flip `state` — that is the
    difference between the monthly rule that was backtested and a daily one.
    """
    months = _rising(10)  # ends June 2026, well above its average
    series = _daily_series(months)
    # Bolt a collapsing July onto the end.
    july = _daily_series([((2026, 7), 1.0)])
    signal = st.evaluate_trend_signal(
        pd.concat([series, july]), sma_months=10, today=date(2026, 7, 15)
    )

    assert signal["state"] == "in"  # unchanged by July
    assert signal["decision_date"].startswith("2026-06")
    assert signal["governs_month"] == "2026-07"
    # The provisional reading *does* see the crash, and is flagged as diverging.
    assert signal["provisional_state"] == "out"
    assert signal["would_flip"] is True


def test_moving_average_excludes_the_running_month():
    """The average is of completed month-ends only, never the partial month."""
    months = _rising(10, start=100.0, step=0.0)  # ten flat month-ends at 100
    series = _daily_series(months)
    july = _daily_series([((2026, 7), 500.0)])  # a spike that must not be averaged in
    signal = st.evaluate_trend_signal(
        pd.concat([series, july]), sma_months=10, today=date(2026, 7, 20)
    )
    assert signal["sma"] == pytest.approx(100.0)


def test_refuses_to_answer_without_enough_history():
    """A partial average biased by its own short window is worse than no answer."""
    series = _daily_series(_rising(6))
    assert (
        st.evaluate_trend_signal(series, sma_months=10, today=date(2026, 7, 5)) is None
    )


def test_out_state_when_below_the_average():
    months = _rising(10, start=200.0, step=-10.0)  # falling into June
    signal = st.evaluate_trend_signal(
        _daily_series(months), sma_months=10, today=date(2026, 7, 10)
    )
    assert signal["state"] == "out"


# --- the flip threshold -----------------------------------------------------


def test_flip_close_is_the_price_that_equals_its_own_average():
    """
    `flip_close` must satisfy the fixed point: a month-end at that price makes
    the close exactly equal the average that includes it.
    """
    months = _rising(10)
    signal = st.evaluate_trend_signal(
        _daily_series(months), sma_months=10, today=date(2026, 7, 8)
    )
    flip = signal["flip_close"]

    completed = [close for (_ym, close) in months]
    window = completed[-9:] + [flip]
    assert flip == pytest.approx(sum(window) / len(window))


def test_distance_pct_is_signed_from_the_flip_price():
    months = _rising(10)
    series = _daily_series(months)
    # Current month trading well above the flip level.
    series = pd.concat([series, _daily_series([((2026, 7), 1000.0)])])
    signal = st.evaluate_trend_signal(series, sma_months=10, today=date(2026, 7, 8))
    assert signal["distance_pct"] > 0
    assert signal["latest_close"] == pytest.approx(1000.0)


# --- strategy definitions ---------------------------------------------------


def test_every_strategy_has_sleeves_that_sum_to_one():
    for strategy in st.list_strategies():
        assert sum(strategy.sleeves.values()) == pytest.approx(1.0), strategy.id


def test_default_strategy_exists():
    assert st.get_strategy(st.DEFAULT_STRATEGY_ID) is not None


def test_unknown_strategy_returns_none():
    assert st.get_strategy("does_not_exist") is None


# --- the constraints: no leverage, no funds ---------------------------------


def test_no_sleeve_type_can_express_leverage_or_a_fund():
    """
    The constraints are structural, not defaults.

    `RankingSleeve` is the only sleeve kind, and it describes a list of ranked
    companies. There is nothing to set a multiplier on and nothing to name a
    fund with, which is the point: a `leverage = 1.0` default or an unused
    `safe_symbol` would keep those code paths alive and one edit from use.
    """
    assert not hasattr(st, "TrendSleeve")
    for field in ("leverage", "index_symbol", "safe_symbol", "risk_symbol"):
        assert not hasattr(st.RankingSleeve, field), field
        assert not hasattr(st.Strategy, field), field


def test_every_strategy_is_stock_only():
    """Positions must be common stock — never a fund, never a cash proxy."""
    for strategy in st.list_strategies():
        assert strategy.ranking is not None, strategy.id
        assert strategy.sleeves == {"ranking": 1.0}, strategy.id


def test_no_strategy_names_a_fund_ticker():
    """
    Named explicitly, because a regression would reintroduce these by name.

    Covers the levered ETFs the first version held, the plain index funds the
    second version held, and the bill ETFs that stood in for cash.
    """
    banned = {
        "QLD",
        "TQQQ",
        "SSO",
        "UPRO",
        "SPXL",  # levered
        "QQQ",
        "SPY",
        "VOO",
        "IVV",
        "VTI",
        "DIA",  # index funds
        "SGOV",
        "BIL",
        "SHV",  # cash proxies
    }
    for strategy in st.list_strategies():
        payload = st.strategy_payload(strategy)
        blob = repr(payload)
        for ticker in banned:
            # Word-boundary match so "SPY" does not fire on prose.
            assert not re.search(rf"\b{ticker}\b", blob), (
                f"{strategy.id} mentions {ticker}"
            )


def test_payload_exposes_no_trend_sleeve():
    for strategy in st.list_strategies():
        payload = st.strategy_payload(strategy)
        assert "trend" not in payload, strategy.id
        assert "leverage" not in repr(payload.get("ranking", {})), strategy.id


def test_allocation_is_a_single_stock_sleeve(monkeypatch):
    """The whole capital goes to ranked companies; no second sleeve exists."""
    monkeypatch.setattr(
        st,
        "_ranking_positions",
        lambda sleeve, capital, today=None: {
            "positions": [
                {
                    "symbol": "AAA",
                    "role": "stock",
                    "weight": 0.5,
                    "amount": capital / 2,
                },
                {
                    "symbol": "BBB",
                    "role": "stock",
                    "weight": 0.5,
                    "amount": capital / 2,
                },
            ],
            "run": {"run_id": 1, "finished_at": "2026-07-28"},
            "error": None,
        },
    )
    allocation = st.build_allocation(st.get_strategy("quality_20"), 100_000.0)

    assert [s["key"] for s in allocation["sleeves"]] == ["ranking"]
    assert allocation["sleeves"][0]["amount"] == pytest.approx(100_000.0)
    assert all(p["role"] == "stock" for p in allocation["sleeves"][0]["positions"])
    # No signal is nested any more — the indicator is not part of a strategy.
    assert "signal" not in allocation


# --- the ranking sleeve's industry cap --------------------------------------


def _score_frame():
    """Six companies: four insurers (SIC 6351) ranked top, then two others."""
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "cik": [1, 2, 3, 4, 5, 6],
            "name": ["A", "B", "C", "D", "E", "F"],
            "quality_score": [90.0, 89.0, 88.0, 87.0, 60.0, 59.0],
            "value_score": [90.0, 89.0, 88.0, 87.0, 60.0, 59.0],
            "confidence": [1.0] * 6,
            "price": [10.0] * 6,
        }
    )


def test_industry_cap_skips_past_a_full_group(monkeypatch):
    """
    The cap fills a slot from further down rather than shrinking the book.

    Without it the top four insurers take 4/5 of a five-name sleeve — one
    balance-sheet risk wearing four tickers.
    """
    frame = _score_frame()
    sic = {str(i).zfill(10): 6351 for i in range(1, 5)}
    sic["0000000005"] = 7372
    sic["0000000006"] = 6022
    monkeypatch.setattr("edgar_sic.get_sic_map", lambda: sic)

    sleeve = st.RankingSleeve(
        quality_weight=0.8, top_n=4, max_per_sector=2, sector_digits=2
    )
    picked = st._apply_sector_cap(frame, frame["quality_score"], sleeve)

    assert list(picked["symbol"]) == ["AAA", "BBB", "EEE", "FFF"]


def test_uncapped_sleeve_keeps_the_raw_order(monkeypatch):
    frame = _score_frame()
    monkeypatch.setattr("edgar_sic.get_sic_map", lambda: {})
    sleeve = st.RankingSleeve(quality_weight=0.8, top_n=3, max_per_sector=None)
    picked = st._apply_sector_cap(frame, frame["quality_score"], sleeve)
    assert list(picked["symbol"]) == ["AAA", "BBB", "CCC"]


def test_unclassified_companies_do_not_cap_each_other(monkeypatch):
    """
    A missing SIC code must not lump every unclassified name into one group.

    Otherwise a data gap, not a shared risk, is what limits the book.
    """
    frame = _score_frame()
    monkeypatch.setattr("edgar_sic.get_sic_map", lambda: {})
    sleeve = st.RankingSleeve(
        quality_weight=0.8, top_n=4, max_per_sector=1, sector_digits=2
    )
    picked = st._apply_sector_cap(frame, frame["quality_score"], sleeve)
    assert len(picked) == 4


def test_sic_map_failure_degrades_to_no_cap(monkeypatch):
    """A missing SIC map must not empty the sleeve."""
    frame = _score_frame()

    def boom():
        raise RuntimeError("sic map unavailable")

    monkeypatch.setattr("edgar_sic.get_sic_map", boom)
    sleeve = st.RankingSleeve(
        quality_weight=0.8, top_n=3, max_per_sector=2, sector_digits=2
    )
    picked = st._apply_sector_cap(frame, frame["quality_score"], sleeve)
    assert len(picked) == 3


# --- live pricing and snapshot staleness ------------------------------------


def _snapshot_frame(count: int = 2):
    """
    A stored ranking run of `count` companies.

    The default of two keeps the `_ranking_positions` tests legible, and those
    pass a matching `top_n`. Anything exercising `build_allocation` against a
    real strategy must ask for twenty, because the rule asks for twenty — a
    two-name snapshot behind a twenty-name rule is a *short book*, which is now
    a warned-about condition rather than a quiet one.
    """
    # The two-name case keeps its original symbols and closes so the
    # price-source tests keep reading the way they were written.
    symbols = ["AAA", "BBB"] if count == 2 else [f"S{i:02d}" for i in range(count)]
    prices = [10.0, 20.0] if count == 2 else [10.0 * (i + 1) for i in range(count)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "cik": list(range(1, count + 1)),
            "name": [f"Company {i}" for i in range(count)],
            "quality_score": [90.0 - i for i in range(count)],
            "value_score": [90.0 - i for i in range(count)],
            "confidence": [1.0] * count,
            "price": prices,  # stale closes stored with the run
        }
    )


def _store(monkeypatch, finished_at="2026-07-28T08:00:00", count: int = 2):
    frame = _snapshot_frame(count)

    class FakeStore:
        def get_run(self, run_id=None):
            return {"run_id": 4, "finished_at": finished_at}

        def get_scores_frame(self, run_id=None):
            return frame

    monkeypatch.setattr("buffett_store.get_store", lambda: FakeStore())
    monkeypatch.setattr("edgar_sic.get_sic_map", lambda: {})


def test_share_counts_use_the_live_quote_not_the_stored_close(monkeypatch):
    """
    Membership comes from the snapshot; the price must not.

    Sizing an order off a stored close hands over a share count that was wrong
    before it was placed, and gets worse the older the snapshot is.
    """
    _store(monkeypatch)
    monkeypatch.setattr(
        st, "latest_closes", lambda symbols, today=None: {"AAA": 20.0, "BBB": 40.0}
    )

    built = st._ranking_positions(
        st.RankingSleeve(quality_weight=0.8, top_n=2, max_per_sector=None), 1000.0
    )
    by_symbol = {p["symbol"]: p for p in built["positions"]}

    # 500 each: at the live 20.0 that is 25 shares, not the 50 the stale 10.0 implies.
    assert by_symbol["AAA"]["price"] == pytest.approx(20.0)
    assert by_symbol["AAA"]["shares"] == 25
    assert by_symbol["BBB"]["shares"] == 12
    assert built["price_source"] == "live"


def test_falls_back_to_the_stored_close_when_a_quote_is_missing(monkeypatch):
    """A quote outage must degrade to the old behaviour, not to no answer."""
    _store(monkeypatch)
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {})

    built = st._ranking_positions(
        st.RankingSleeve(quality_weight=0.8, top_n=2, max_per_sector=None), 1000.0
    )
    by_symbol = {p["symbol"]: p for p in built["positions"]}
    assert by_symbol["AAA"]["price"] == pytest.approx(10.0)  # the stored close
    assert by_symbol["AAA"]["shares"] == 50
    assert built["price_source"] == "snapshot"


def test_partial_quote_coverage_is_reported_as_mixed(monkeypatch):
    _store(monkeypatch)
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {"AAA": 20.0})
    built = st._ranking_positions(
        st.RankingSleeve(quality_weight=0.8, top_n=2, max_per_sector=None), 1000.0
    )
    assert built["price_source"] == "mixed"


def test_a_capped_run_is_not_saved_over_the_production_ranking(monkeypatch):
    """
    `--limit` is a smoke test and must not become the snapshot strategies read.

    Every strategy reads the newest completed run, so a five-filer test run
    that got saved would silently become the book the app tells you to hold.
    That has happened; this is the structural stop.
    """
    import buffett_rank_worker as worker

    seen = {}

    class FakeResult:
        run_id, duration_seconds, stats = None, 0.1, {}

    def fake_run(limit=None, persist=True, skip_market_data=False):
        seen["limit"], seen["persist"] = limit, persist
        return FakeResult()

    monkeypatch.setattr(worker.buffett_pipeline, "run", fake_run)

    assert worker.run_once(limit=5) is True
    assert seen["persist"] is False, "a capped run must not be persisted"

    assert worker.run_once(limit=None) is True
    assert seen["persist"] is True, "a full run must be persisted"

    assert worker.run_once(limit=5, persist_partial=True) is True
    assert seen["persist"] is True, "--persist-partial must override"


def test_short_book_is_warned_about_and_not_silently_repriced(monkeypatch):
    """
    A truncated ranking must not look like a complete answer.

    This is a real incident, not a hypothetical: a five-filer smoke test was
    saved as the newest run and the sleeve served a two-name book against a
    twenty-name rule, allocating a fraction of the capital without a word. The
    weights stay at the rule's 1/N — widening them over a shorter list would
    quietly change the concentration the backtest actually measured.
    """
    _store(monkeypatch, count=5)
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {})

    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1_000_000.0, today=date(2026, 7, 29)
    )
    sleeve = allocation["sleeves"][0]

    assert allocation["is_short"] is True
    assert sleeve["positions_requested"] == 20
    assert sleeve["positions_filled"] == 5
    # Five slots at the rule's 5% each, not five slots at 20%.
    assert sleeve["amount_allocated"] == pytest.approx(250_000.0)
    assert all(p["weight"] == pytest.approx(0.05) for p in sleeve["positions"])

    short = [w for w in allocation["warnings"] if "only 5 of the 20" in w]
    assert len(short) == 1
    assert "$750,000" in short[0]


def test_a_full_book_says_nothing_about_being_short(monkeypatch):
    _store(monkeypatch, count=20)
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {})

    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1_000_000.0, today=date(2026, 7, 29)
    )
    sleeve = allocation["sleeves"][0]
    assert allocation["is_short"] is False
    assert sleeve["positions_filled"] == sleeve["positions_requested"] == 20
    assert sleeve["amount_allocated"] == pytest.approx(1_000_000.0)
    assert allocation["warnings"] == []


def test_an_empty_ranking_is_not_reported_as_merely_short(monkeypatch):
    """No run at all already has its own error; `is_short` would be noise."""
    _store(monkeypatch, count=0)
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {})

    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1_000_000.0, today=date(2026, 7, 29)
    )
    assert allocation["is_short"] is False
    assert not [w for w in allocation["warnings"] if "of the 20 names" in w]


def test_ranking_age_is_measured_in_whole_days():
    assert st.ranking_age_days("2026-07-28T08:00:00", date(2026, 7, 29)) == 1
    assert st.ranking_age_days("2026-07-01T08:00:00", date(2026, 7, 29)) == 28
    # A run stamped in the future is age zero, never negative.
    assert st.ranking_age_days("2026-08-01T08:00:00", date(2026, 7, 29)) == 0
    assert st.ranking_age_days(None) is None
    assert st.ranking_age_days("not-a-date") is None


def test_fresh_snapshot_raises_no_warning(monkeypatch):
    _store(monkeypatch, finished_at="2026-07-28T08:00:00", count=20)
    monkeypatch.setattr(
        st, "latest_closes", lambda symbols, today=None: {"AAA": 20.0, "BBB": 40.0}
    )
    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1000.0, today=date(2026, 7, 29)
    )
    assert allocation["ranking_age_days"] == 1
    assert allocation["ranking_is_stale"] is False
    assert allocation["warnings"] == []


def test_stale_snapshot_warns_because_a_dead_worker_looks_healthy(monkeypatch):
    """
    The endpoint keeps serving the last good run whether or not the batch
    worker is alive, so silence is indistinguishable from health. Past the
    threshold it must say so.
    """
    _store(monkeypatch, finished_at="2026-05-01T08:00:00", count=20)
    monkeypatch.setattr(
        st, "latest_closes", lambda symbols, today=None: {"AAA": 20.0, "BBB": 40.0}
    )
    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1000.0, today=date(2026, 7, 29)
    )
    assert allocation["ranking_is_stale"] is True
    assert allocation["ranking_age_days"] == 89
    assert len(allocation["warnings"]) == 1
    assert "89 days old" in allocation["warnings"][0]
    assert "2026-05-01" in allocation["warnings"][0]


def test_staleness_threshold_boundary(monkeypatch):
    """Exactly at the threshold counts as stale — a week of missed daily runs."""
    monkeypatch.setattr(st, "latest_closes", lambda symbols, today=None: {})
    for age, expected in (
        (st.STALE_RANKING_DAYS - 1, False),
        (st.STALE_RANKING_DAYS, True),
    ):
        _store(
            monkeypatch,
            finished_at=str(date(2026, 7, 29) - timedelta(days=age)),
            count=20,
        )
        allocation = st.build_allocation(
            st.get_strategy("quality_20"), 1000.0, today=date(2026, 7, 29)
        )
        assert allocation["ranking_is_stale"] is expected, age


def test_sleeve_reports_where_its_prices_came_from(monkeypatch):
    _store(monkeypatch)
    monkeypatch.setattr(
        st, "latest_closes", lambda symbols, today=None: {"AAA": 20.0, "BBB": 40.0}
    )
    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 1000.0, today=date(2026, 7, 29)
    )
    assert allocation["sleeves"][0]["price_source"] == "live"


# --- serialisation ----------------------------------------------------------


def test_market_signal_is_self_describing_and_marked_advisory(monkeypatch):
    """
    The indicator must carry its own symbol and an explicit advisory flag.

    The flag is not decoration: no strategy acts on this signal, and a client
    that presented it as a trading instruction would be claiming backing the
    measurements do not give it.
    """
    series = _daily_series(_rising(12))
    monkeypatch.setattr(st, "load_signal_prices", lambda symbol, today=None: series)

    signal = st.market_trend_signal("QQQ", 10, today=date(2026, 7, 10))

    assert signal["signal_symbol"] == "QQQ"
    assert signal["advisory_only"] is True
    # The panel shows several of these at once; the name and the zone travel
    # with each reading so no client has to invent either.
    assert signal["signal_name"] == "NASDAQ 100"
    assert signal["market_timezone"] == st.MARKET_SIGNAL_TIMEZONE


def test_every_panel_index_is_named(monkeypatch):
    """
    Each market the panel reads must produce a reading that names itself.

    Falling back to the ticker would put "SPY" where the reader expects
    "S&P 500", so the display-name map has to cover the whole set — and cover it
    here, rather than in three clients each keeping their own copy.
    """
    series = _daily_series(_rising(12))
    monkeypatch.setattr(st, "load_signal_prices", lambda symbol, today=None: series)

    assert st.MARKET_SIGNAL_INDICES, "the panel must read at least one market"
    for symbol, label in st.MARKET_SIGNAL_INDICES:
        signal = st.market_trend_signal(symbol, 10, today=date(2026, 7, 10))
        assert signal["signal_name"] == label != symbol, symbol


def test_the_running_month_is_decided_on_the_market_clock(monkeypatch):
    """
    Which month is still running is a US market fact, not a server one.

    Investa's server runs on a Bangkok clock, up to a day ahead of New York. On
    1 August in Bangkok it is still 31 July in New York, and July's final
    session has yet to close: reading the server date would retire July's
    month-end a day early and hand the panel a decision the market never made.

    Pinned by the seam rather than by a story about today — the market clock is
    stubbed to a date in the past, so a regression to `date.today()` cannot
    coincide with it in any month this test is ever run.
    """
    series = _daily_series(_rising(12, end_year=2024, end_month=4))
    monkeypatch.setattr(st, "load_signal_prices", lambda symbol, today=None: series)
    monkeypatch.setattr(st, "get_est_today", lambda: date(2024, 5, 10))

    signal = st.market_trend_signal("SPY", 10)

    assert signal["governs_month"] == "2024-05"
    assert signal["decision_date"].startswith("2024-04")
    assert signal["next_decision_date"] == "2024-05-31"


def test_allocation_amounts_sum_to_the_capital_given(monkeypatch):
    monkeypatch.setattr(
        st,
        "_ranking_positions",
        lambda sleeve, capital, today=None: {
            "positions": [],
            "run": None,
            "error": None,
        },
    )
    allocation = st.build_allocation(
        st.get_strategy("quality_20"), 100_000.0, today=date(2026, 7, 10)
    )
    assert sum(s["amount"] for s in allocation["sleeves"]) == pytest.approx(100_000.0)


def test_payload_is_json_safe_and_complete():
    payload = st.strategy_payload(st.get_strategy("quality_20"))
    assert payload["id"] == "quality_20"
    assert payload["is_default"] is True
    assert payload["ranking"]["max_per_sector"] == 3
    assert payload["ranking"]["top_n"] == 20
    assert payload["backtest"]["cagr"] > 0
    assert payload["risks"]


def test_payload_always_describes_its_ranking_sleeve():
    for strategy in st.list_strategies():
        payload = st.strategy_payload(strategy)
        assert payload["ranking"]["top_n"] > 0, strategy.id
        assert "min_market_cap" in payload["ranking"], strategy.id


def test_all_expected_strategies_exist():
    expected_ids = {
        "quality_20",
        "quality_15",
        "quality_20_uncapped",
        "quality_largecap_20",
        "quality_value_balanced",
        "quality_pure",
    }
    registered_ids = {s.id for s in st.list_strategies()}
    assert registered_ids == expected_ids


def test_min_market_cap_filters_positions(monkeypatch):
    """Companies below min_market_cap must be excluded before picking."""
    frame = pd.DataFrame(
        {
            "symbol": ["SMALL", "LARGE"],
            "cik": [1, 2],
            "name": ["Small Co", "Large Co"],
            "quality_score": [95.0, 85.0],
            "value_score": [95.0, 85.0],
            "market_cap": [5_000_000_000.0, 50_000_000_000.0],
            "confidence": [1.0, 1.0],
            "price": [10.0, 20.0],
        }
    )

    class FakeStore:
        def get_run(self, run_id=None):
            return {"run_id": 1, "finished_at": "2026-07-28T08:00:00"}

        def get_scores_frame(self, run_id=None):
            return frame

    monkeypatch.setattr("buffett_store.get_store", lambda: FakeStore())
    monkeypatch.setattr("edgar_sic.get_sic_map", lambda: {})
    monkeypatch.setattr(
        st, "latest_closes", lambda symbols, today=None: {"LARGE": 20.0}
    )

    sleeve = st.RankingSleeve(
        quality_weight=0.8,
        top_n=1,
        max_per_sector=3,
        min_market_cap=10_000_000_000.0,
    )
    res = st._ranking_positions(sleeve, 10_000.0)
    assert len(res["positions"]) == 1
    assert res["positions"][0]["symbol"] == "LARGE"
