"""Tests for the Bank of Thailand FX provider.

The gateway needs a bearer token, so every network call here is stubbed. What is
under test is the handful of things that would otherwise go wrong quietly:

*The reciprocal.* The BOT quotes baht per unit of foreign currency — the
transpose of the ECB's orientation, and of what `fx_pairs` expects. Miss it and
`THB=X` reads 0.031 instead of 32.7: a number that stores cleanly, looks like a
rate, and values a Thai holding a thousandfold wrong.

*The empty row.* Asking for a period before the series starts returns HTTP 200
with one `data_detail` entry whose `period` is `''`. Count rows and every month
of the 1990s looks like it has a day of data in it.

*The limits.* 31 days a request, enforced with a 400; and a quota that answers
429 with no `Retry-After` and no `X-RateLimit-*` header to pace against.
"""

import os
import sys
from datetime import date

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "scripts"))

import bot_fx_provider as bot  # noqa: E402

# One real row, 25 Aug 2026.
ROW = {
    "period": "2026-08-25",
    "currency_id": "USD",
    "currency_name_eng": "USA : DOLLAR (USD) ",
    "buying_sight": "32.4714000",
    "buying_transfer": "32.5611000",
    "selling": "32.8942000",
    "mid_rate": "32.7277000",
}

# What the gateway sends for a range it has nothing for.
EMPTY_ROW = {"period": "", "currency_id": "", "mid_rate": None}


def payload(rows):
    return {"result": {"data": {"data_header": {}, "data_detail": rows}}}


class FakeResponse:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text or ""
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    """Replays queued responses and records the params it was asked for."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0) if self.responses else FakeResponse(200, payload([]))


@pytest.fixture
def provider(monkeypatch):
    p = bot.BOTFXProvider(api_key="test-token")
    monkeypatch.setattr(bot.time, "sleep", lambda *_: None)
    return p


def with_session(provider, responses):
    session = FakeSession(responses)
    provider._session = session
    return session


# --- orientation -----------------------------------------------------------


def test_the_reciprocal_is_taken_so_a_baht_cross_is_not_inverted():
    rates = bot.parse_detail([ROW])
    # Stored internally as USD per baht...
    assert rates["2026-08-25"]["USD"] == pytest.approx(1 / 32.7277)
    # ...so that the pair reads as baht per dollar, which is what THB=X means.
    assert bot.pair_rate(rates["2026-08-25"], "THB=X") == pytest.approx(32.7277)
    assert bot.pair_rate(rates["2026-08-25"], "USDTHB=X") == pytest.approx(32.7277)
    assert bot.pair_rate(rates["2026-08-25"], "THBUSD=X") == pytest.approx(1 / 32.7277)


def test_baht_per_undoes_the_reciprocal_for_reporting():
    rates = bot.parse_detail([ROW])
    assert bot.baht_per(rates, "2026-08-25", "USD") == pytest.approx(32.7277)
    assert bot.baht_per(rates, "2026-08-25", "XXX") is None


def test_the_mid_rate_is_the_price_not_the_commercial_spread():
    """buying_sight/selling are what a bank charges, not what a position is worth."""
    rates = bot.parse_detail([ROW])
    quoted = bot.baht_per(rates, "2026-08-25", "USD")
    assert quoted == pytest.approx(32.7277)
    assert quoted != pytest.approx(float(ROW["buying_sight"]))
    assert quoted != pytest.approx(float(ROW["selling"]))


# --- the empty row ---------------------------------------------------------


def test_a_period_less_row_is_not_a_day_of_data():
    assert bot.parse_detail([EMPTY_ROW]) == {}
    assert bot.parse_detail([EMPTY_ROW, ROW]) != {}
    assert len(bot.parse_detail([EMPTY_ROW, ROW])) == 1


@pytest.mark.parametrize("bad", ["", None, "-", "not a number", "0"])
def test_an_unusable_mid_rate_is_dropped(bad):
    assert bot.parse_detail([{**ROW, "mid_rate": bad}]) == {}


# --- request windows -------------------------------------------------------


def test_windows_never_exceed_what_the_gateway_accepts():
    windows = bot.month_windows(date(2002, 1, 1), date(2002, 12, 31))
    assert all(
        (stop - start).days < bot.MAX_PERIOD_DAYS for start, stop in windows
    ), "a wider range is a 400, not a truncated result"
    assert windows[0][0] == date(2002, 1, 1)
    assert windows[-1][1] == date(2002, 12, 31)
    # Contiguous, no gap and no overlap.
    for (_, stop), (nxt, _) in zip(windows, windows[1:]):
        assert (nxt - stop).days == 1


def test_a_request_before_the_series_starts_is_clamped(provider):
    session = with_session(provider, [FakeResponse(200, payload([ROW]))])
    provider.fetch_daily_avg(date(1998, 1, 1), date(2002, 1, 20), throttle=0)
    assert session.calls[0][1]["start_period"] == bot.SERIES_START.isoformat()
    assert len(session.calls) == 1, "the 1998-2001 windows must not be requested"


def test_currency_none_asks_for_every_currency(provider):
    session = with_session(provider, [FakeResponse(200, payload([ROW]))])
    provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), currency=None, throttle=0)
    assert "currency" not in session.calls[0][1]


# --- failures --------------------------------------------------------------


def test_no_token_is_its_own_error(monkeypatch):
    import config

    monkeypatch.setattr(config, "BOT_API_KEY", None)
    p = bot.BOTFXProvider(api_key=None)
    assert not p.is_configured()
    with pytest.raises(bot.BOTFXNotConfiguredError):
        p.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20))


def test_a_403_says_the_app_is_not_entitled_rather_than_unauthorised(provider):
    """The 401/403 split is the only signal separating these two, and the
    difference decides whether you go fix the token or go fix the subscription."""
    with_session(provider, [FakeResponse(403, text='{"error":"disallowed"}')])
    with pytest.raises(bot.BOTFXError) as exc:
        provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), throttle=0)
    assert exc.value.status == 403
    assert "not approved" in str(exc.value)


def test_a_429_is_retried_and_then_reported(provider):
    limited = [FakeResponse(429, text="Rate Limit Exceeded")] * (
        bot.RATE_LIMIT_RETRIES + 1
    )
    session = with_session(provider, limited)
    with pytest.raises(bot.BOTFXError) as exc:
        provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), throttle=0)
    assert exc.value.status == 429
    assert len(session.calls) == bot.RATE_LIMIT_RETRIES + 1


def test_a_429_that_clears_is_invisible_to_the_caller(provider):
    session = with_session(
        provider,
        [FakeResponse(429, text="Rate Limit Exceeded"), FakeResponse(200, payload([ROW]))],
    )
    rates = provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), throttle=0)
    assert len(session.calls) == 2
    assert bot.baht_per(rates, "2026-08-25", "USD") == pytest.approx(32.7277)


def test_a_run_that_cannot_fit_the_budget_is_refused_before_it_spends_anything(
    provider,
):
    """200 calls/hour on a rolling window: discovering that 200 calls in leaves
    the work half done and nothing left to finish it with."""
    session = with_session(provider, [FakeResponse(200, payload([ROW]))] * 400)
    with pytest.raises(bot.BOTFXError) as exc:
        # The full span is ~290 windows against a budget of 200.
        provider.fetch_daily_avg(bot.SERIES_START, date(2026, 8, 26), throttle=0)
    assert exc.value.status == 429
    assert session.calls == [], "it must refuse before making a single call"
    assert "resumable" in str(exc.value)


def test_the_budget_counts_only_the_rolling_window(provider, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(bot.time, "monotonic", lambda: now[0])
    provider._call_times.extend([now[0]] * 50)
    assert provider.calls_this_hour() == 50
    assert provider.budget_remaining() == bot.RATE_LIMIT_PER_HOUR - 50

    # An hour later those calls have aged out and the budget is whole again.
    now[0] += bot.RATE_LIMIT_WINDOW_SECONDS + 1
    assert provider.calls_this_hour() == 0
    assert provider.budget_remaining() == bot.RATE_LIMIT_PER_HOUR


def test_every_attempt_counts_against_the_budget_including_a_429(provider):
    """A rejected call still reached the gateway; pretending otherwise is how a
    retry loop convinces itself it has headroom it does not have."""
    with_session(
        provider,
        [FakeResponse(429, text="Rate Limit Exceeded"), FakeResponse(200, payload([ROW]))],
    )
    provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), throttle=0)
    assert provider.calls_this_hour() == 2


def test_a_non_json_body_is_an_error_not_a_crash(provider):
    with_session(provider, [FakeResponse(200, None, text="<html>gateway</html>")])
    with pytest.raises(bot.BOTFXError):
        provider.fetch_daily_avg(date(2026, 8, 1), date(2026, 8, 20), throttle=0)


# --- only asking for what is missing ---------------------------------------


def test_only_windows_holding_a_missing_day_are_requested(tmp_path):
    import backfill_fx_rates as job
    from market_db import MarketDatabase

    path = str(tmp_path / "market_data.db")
    db = MarketDatabase(path)
    # Every business day of January stored; February entirely absent.
    day = date(2002, 1, 1)
    rows = []
    while day <= date(2002, 1, 31):
        if day.weekday() < 5:
            rows.append((day.isoformat(), 43.0))
        day += timedelta_days(1)
    for pair in job.BOT_DEFAULT_PAIRS:
        db.upsert_fx_rows(pair, rows)

    windows = job.missing_windows(
        path, job.BOT_DEFAULT_PAIRS, date(2002, 1, 1), date(2002, 2, 28)
    )

    assert windows, "February is missing entirely"
    assert all(stop >= date(2002, 2, 1) for _, stop in windows), (
        "January is complete and must not be re-requested"
    )


def test_a_missing_weekend_does_not_pull_a_window_into_the_run(tmp_path):
    """The BOT publishes business days only, so a Saturday can never be filled."""
    import backfill_fx_rates as job
    from market_db import MarketDatabase

    path = str(tmp_path / "market_data.db")
    db = MarketDatabase(path)
    day = date(2026, 8, 3)  # Monday
    rows = []
    while day <= date(2026, 8, 7):
        rows.append((day.isoformat(), 32.5))
        day += timedelta_days(1)
    for pair in job.BOT_DEFAULT_PAIRS:
        db.upsert_fx_rows(pair, rows)

    # 8-9 Aug is the weekend, and nothing else in the range is absent.
    assert (
        job.missing_windows(
            path, job.BOT_DEFAULT_PAIRS, date(2026, 8, 3), date(2026, 8, 9)
        )
        == []
    )


def timedelta_days(n):
    from datetime import timedelta

    return timedelta(days=n)
