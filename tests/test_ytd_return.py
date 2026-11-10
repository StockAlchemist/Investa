"""YTD return: one figure, from one helper, in the summary and risk metrics.

The summary's `ytd_return` was declared on every client and never set, so the
web Performance strip always took its fallback, which ADDED the year's monthly
returns: +9.92% for a 2026 whose annual bar and compounded months both read
+7.25%. The summary now ships the time-weighted YTD itself, from the same
helper `/risk_metrics` uses, so the two cannot disagree.
"""

import asyncio
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from server.auth import User
from server import portfolio_service as service
from server.portfolio_service import ytd_twr_return
from server.routes import portfolio as routes

TODAY = date(2026, 11, 10)


def _history(points):
    """A daily history with a TWR index, keyed by calendar day."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in points], name="Date")
    return pd.DataFrame(
        {"Portfolio Accumulated Gain": [v for _, v in points]}, index=idx
    )


# --- The helper ---------------------------------------------------------------


def test_measured_from_the_last_session_of_the_previous_year():
    daily = _history(
        [
            ("2025-06-30", 1.10),
            ("2025-12-31", 1.20),
            ("2026-01-02", 1.25),
            ("2026-11-09", 1.287),
        ]
    )
    assert ytd_twr_return(daily, TODAY) == pytest.approx(1.287 / 1.20 - 1)


def test_an_account_opened_this_year_is_measured_from_its_first_point():
    daily = _history([("2026-03-02", 1.0), ("2026-11-09", 1.05)])
    assert ytd_twr_return(daily, TODAY) == pytest.approx(0.05)


def test_the_year_is_the_market_year_passed_in():
    """On 1 Jan in Bangkok it is still 31 Dec in New York: 2026 is not over."""
    daily = _history([("2025-12-31", 1.20), ("2026-12-31", 1.30)])
    assert ytd_twr_return(daily, date(2026, 12, 31)) == pytest.approx(1.30 / 1.20 - 1)


def test_a_string_index_is_read_as_dates():
    daily = _history([("2025-12-31", 1.20), ("2026-11-09", 1.32)])
    daily.index = daily.index.strftime("%Y-%m-%d")
    assert ytd_twr_return(daily, TODAY) == pytest.approx(0.10)


def test_it_does_not_rewrite_the_shared_frame():
    daily = _history([("2025-12-31", 1.20), ("2026-11-09", 1.32)])
    daily.index = daily.index.strftime("%Y-%m-%d")
    ytd_twr_return(daily, TODAY)
    assert daily.index[0] == "2025-12-31"


@pytest.mark.parametrize(
    "daily",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame(
            {"Portfolio Value": [1.0]}, index=pd.DatetimeIndex(["2026-01-02"])
        ),
        _history([("2025-12-31", 0.0), ("2026-11-09", 1.1)]),
        _history([("2025-12-31", float("nan")), ("2026-11-09", 1.1)]),
        _history([("2025-12-31", 1.2), ("2026-11-09", float("nan"))]),
    ],
    ids=["none", "empty", "no-twr-column", "zero-base", "nan-base", "nan-latest"],
)
def test_unusable_history_gives_none(daily):
    assert ytd_twr_return(daily, TODAY) is None


# --- The summary ships it -----------------------------------------------------


@pytest.fixture
def portfolio(monkeypatch):
    service.clear_portfolio_caches()
    manager = SimpleNamespace(
        manual_overrides={}, gui_config={}, load_manual_overrides=lambda: None
    )
    monkeypatch.setattr(service, "get_config_manager", lambda _: manager)
    monkeypatch.setattr(routes, "get_config_manager", lambda _: manager)
    monkeypatch.setattr(service, "get_mdp", lambda: None)
    monkeypatch.setattr(service, "is_market_open", lambda: False)
    monkeypatch.setattr(service, "get_est_today", lambda: TODAY)
    df = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01"]), "Account": ["A"]})
    rows = pd.DataFrame([{"Symbol": "OPEN", "Account": "A", "Quantity": 2}])
    result = (
        {"market_value": 100, "dividends": 0, "total_buy_cost": 100},
        rows,
        {},
        {"A": {"market_value": 100}},
        None,
        None,
        None,
    )
    monkeypatch.setattr(service, "calculate_portfolio_summary", lambda **_: result)
    data = (df, {}, {}, set(), {}, {}, "/isolated-ytd-return.db", 0)
    yield data, User(id=1, username="ytd-test", created_at="2026-01-01")
    service.clear_portfolio_caches()


def test_summary_ships_the_ytd_twr_in_percent(portfolio, monkeypatch):
    data, user = portfolio
    daily = _history([("2025-01-02", 1.0), ("2025-12-31", 1.20), ("2026-11-09", 1.287)])
    monkeypatch.setattr(
        service,
        "calculate_historical_performance",
        lambda **_: (daily, {}, {}, "|||TWR_FACTOR:1.287"),
    )

    async def scenario():
        summary = await service._calculate_portfolio_summary_internal(
            data=data, current_user=user
        )
        if summary["freshness"]["refreshing"]:
            await asyncio.gather(
                *list(service._PORTFOLIO_HISTORY_CACHE._inflight.values())
            )
            summary = await service._calculate_portfolio_summary_internal(
                data=data, current_user=user
            )
        return summary

    metrics = asyncio.run(scenario())["metrics"]
    assert metrics["ytd_return"] == pytest.approx((1.287 / 1.20 - 1) * 100)
    assert metrics["cumulative_twr"] == pytest.approx(28.7)
