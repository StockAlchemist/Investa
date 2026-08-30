# -*- coding: utf-8 -*-
"""Bank of Thailand — the official THB rate (plan Phase 5.1, second half).

The ECB's reference rates only carry THB from **2005-04-01**, the day it joined
the list. Investa's ledger opens 29 Jun 2002 and its base currency is the baht,
so the first three years of every US holding are converted at a rate that came
from Yahoo or, where Yahoo had nothing, from the previous rate carried forward.
The BOT publishes the real one from **2002-01-02**, which covers the ledger
entirely.

    GET /Stat-ExchangeRate/v2/DAILY_AVG_EXG_RATE/
        ?start_period=yyyy-MM-dd&end_period=yyyy-MM-dd[&currency=USD]

Rates are quoted **baht per one unit of foreign currency**, so this table's unit
is THB and every pair is a cross through the baht (see `fx_pairs`). `mid_rate`
is the one to store: `buying_sight`, `buying_transfer` and `selling` are the
commercial spread around it, not the rate a position is worth.

Credentials: a bearer token from https://portal.api.bot.or.th, issued **per
subscription and not per app** — an app that has not been approved for the
Exchange Rates product returns 403 with a token that is otherwise perfectly
valid. `BOT_API_KEY=…` in `.env`.

Three things that will bite whoever comes next, all verified against the live
gateway on 26 Aug 2026:

  * **The gateway moved and nearly every write-up online is stale.**
    `apigw1.bot.or.th/bot/public` was retired on 31 Dec 2025 and no longer
    resolves in DNS at all; so does the old portal `apiportal.bot.or.th`. The
    new gateway is `gateway.api.bot.or.th` and the header changed from
    `X-IBM-Client-Id: <id>` to `Authorization: Bearer <token>`.
  * **31 days per request, hard.** A wider range is a 400, not a truncated
    result, so history comes back a month at a time — about 290 requests for the
    full span, against a budget of 200 an hour.
  * **200 calls per hour, and the hour rolls.** Total volume is unlimited; the
    rate is not. Because the window is rolling rather than resetting on the
    hour, a backoff is close to useless — nothing frees up until the oldest call
    ages out, so retrying for a couple of minutes just burns attempts. The
    answer is to spend less: `scripts/backfill_fx_rates.py` asks only for the
    windows holding a day the archive lacks (28 rather than 290 for the first
    fill, none on an ordinary night), and `check_budget` refuses a run that
    cannot fit rather than discovering it 200 calls in with the work half done.
  * **No data is a row, not an empty list.** Ask for 2001 and the response is
    HTTP 200 carrying one `data_detail` entry whose `period` is `''`. Take the
    row count as a measure of coverage and every pre-2002 month looks populated.

    python src/bot_fx_provider.py verify
    python src/bot_fx_provider.py rates --currency USD --start 2002-01-01 --end 2002-02-28
"""

import logging
import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Deque, Dict, List, Optional

import requests

import config
from fx_pairs import RateTable
from fx_pairs import pair_rate as _pair_rate
from fx_pairs import pair_series as _pair_series
from fx_pairs import supported_pairs as _supported_pairs

logger = logging.getLogger(__name__)

BOT_GATEWAY = "https://gateway.api.bot.or.th"
DAILY_AVG_PATH = "/Stat-ExchangeRate/v2/DAILY_AVG_EXG_RATE/"

DEFAULT_TIMEOUT = 60

# The gateway rejects a wider window with a 400 rather than truncating.
MAX_PERIOD_DAYS = 31

# Rates are quoted in baht per unit of foreign currency, so the baht is the
# table's own unit and never appears as a currency in a row.
UNIT = "THB"

SOURCE = "bot"

# The daily average series does not reach further back than this; earlier months
# answer 200 with a single empty-period row.
SERIES_START = date(2002, 1, 2)

# The published limit: 200 calls per hour, unlimited total. The hour is a
# **rolling** window, which is the part that decides how this client behaves.
#
# It means a backoff is close to useless. Once the budget is spent, nothing
# clears until the oldest call ages out, so retrying for a couple of minutes —
# the shape that works against a per-second limiter — just burns attempts and
# then reports failure. What works is not spending the budget in the first
# place: the ingester requests only the windows holding a day the archive
# lacks (28 rather than 291 for the first fill), and this client refuses to
# start a run it knows cannot fit.
RATE_LIMIT_PER_HOUR = 200
RATE_LIMIT_WINDOW_SECONDS = 3600

# Courtesy gap between calls. Well inside the limit at 200/hour; it exists so a
# burst does not arrive as a burst.
THROTTLE_SECONDS = 0.4

# Retries are for the boundary case only — a call about to roll out of the
# window — so they are few, and the message says what the real constraint is.
RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_WAIT = 90.0


class BOTFXError(Exception):
    """The gateway refused the request or answered something unreadable."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class BOTFXNotConfiguredError(BOTFXError):
    """No token. Distinct because it is a setup problem, not an outage."""


def pair_rate(row: Dict[str, float], pair: str) -> Optional[float]:
    """This feed's rate for one day's row — see fx_pairs.pair_rate."""
    return _pair_rate(row, pair, UNIT)


def pair_series(rates: RateTable, pair: str):
    """(date, rate) for every day the pair is derivable, oldest first."""
    return _pair_series(rates, pair, UNIT)


def supported_pairs(rates: RateTable, candidates):
    """Those candidates the feed can actually price on at least one day."""
    return _supported_pairs(rates, candidates, UNIT)


def parse_detail(rows: List[dict]) -> RateTable:
    """`data_detail` -> a RateTable in `fx_pairs`' orientation.

    **The reciprocal is taken here, and it has to be.** A `RateTable` means
    "units of this currency per one unit of the table's own unit" — the ECB
    publishes exactly that (USD per EUR), and the BOT publishes its transpose
    (baht per USD, its own unit on the wrong side of the division). Left as
    served, every cross comes out inverted: `THB=X` reads 0.031 instead of 32.7,
    which is a number that stores cleanly, looks like a rate, and values a Thai
    holding a thousandfold wrong.

    Converting at this edge is what lets one set of pair helpers serve both
    feeds. `mid_rate` is the source figure — the commercial buying and selling
    columns are the spread around it, not what a position is worth.

    A row whose `period` is empty is the gateway's way of saying "nothing for
    that range"; it is dropped rather than counted, because otherwise every
    month before 2002 reads as one day of coverage.
    """
    rates: RateTable = {}
    for row in rows or []:
        day = (row.get("period") or "").strip()
        code = (row.get("currency_id") or "").strip().upper()
        raw = row.get("mid_rate")
        if not day or not code or raw in (None, "", "-"):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            rates.setdefault(day, {})[code] = 1.0 / value
    return rates


def baht_per(rates: RateTable, day: str, currency: str) -> Optional[float]:
    """The figure as the BOT published it, undoing `parse_detail`'s reciprocal."""
    value = rates.get(day, {}).get(currency.upper())
    return (1.0 / value) if value else None


def month_windows(start: date, end: date) -> List[tuple]:
    """Split a range into windows the gateway will accept."""
    windows = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=MAX_PERIOD_DAYS - 1), end)
        windows.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return windows


class BOTFXProvider:
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        # Read through the module so a key saved from Settings takes effect
        # now: `from config import BOT_API_KEY` would bind a copy at import
        # time that no later update could reach.
        self.api_key = api_key if api_key is not None else config.BOT_API_KEY
        self.timeout = timeout
        self._session = None
        # When this process made each call, so it can say how much of the
        # rolling budget it is itself responsible for. It cannot see calls made
        # by anything else holding the same token, so this is a floor on usage,
        # never a guarantee of headroom.
        self._call_times: Deque[float] = deque()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def calls_this_hour(self) -> int:
        """How many calls this process has made inside the rolling window."""
        cutoff = time.monotonic() - RATE_LIMIT_WINDOW_SECONDS
        while self._call_times and self._call_times[0] < cutoff:
            self._call_times.popleft()
        return len(self._call_times)

    def budget_remaining(self) -> int:
        return max(0, RATE_LIMIT_PER_HOUR - self.calls_this_hour())

    def check_budget(self, requests_needed: int) -> None:
        """Refuse a run that cannot fit, before it spends anything.

        Discovering the limit 200 calls in leaves the work half done and the
        budget empty for the next hour; saying so up front leaves both intact.
        """
        remaining = self.budget_remaining()
        if requests_needed > remaining:
            raise BOTFXError(
                f"This run needs {requests_needed} call(s) and {remaining} of the "
                f"{RATE_LIMIT_PER_HOUR}/hour budget are left "
                f"({self.calls_this_hour()} already used by this process). Narrow "
                "the range with --start/--end and run it again in the next hour — "
                "the fill is resumable, so nothing is repeated.",
                status=429,
            )

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "accept": "application/json",
                }
            )
        return self._session

    def _get(self, path: str, params: Dict[str, object]) -> dict:
        if not self.is_configured():
            raise BOTFXNotConfiguredError(
                "No BOT_API_KEY. Subscribe an app to the Exchange Rates product at "
                "https://portal.api.bot.or.th and copy the token from the app page."
            )
        # 429 is not hypothetical: a full backfill is ~290 requests, and running
        # two in succession trips the gateway's limiter. It answers with a plain
        # "Rate Limit Exceeded" and usually no Retry-After, so the wait doubles
        # from a second — the same shape as the Yahoo cooldown in market_data.
        delay = 1.0
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                self._call_times.append(time.monotonic())
                response = self._get_session().get(
                    f"{BOT_GATEWAY}{path}", params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise BOTFXError(f"BOT request failed: {exc}") from exc

            if response.status_code != 429:
                break
            if attempt == RATE_LIMIT_RETRIES:
                raise BOTFXError(
                    f"BOT rate limited ({RATE_LIMIT_PER_HOUR} calls/hour, rolling). "
                    f"{self.calls_this_hour()} call(s) from this process are still "
                    "inside the window; it clears as they age out, not on a timer, "
                    "so wait rather than retrying. The backfill is fill-only, so "
                    "re-running resumes without duplicating anything.",
                    status=429,
                )
            wait = float(response.headers.get("Retry-After") or delay)
            logger.warning("BOT rate limited; waiting %.0fs before retry", wait)
            time.sleep(wait)
            delay = min(delay * 2, MAX_RATE_LIMIT_WAIT)

        if response.status_code == 403:
            # A valid token with nothing behind it. Worth saying plainly: the
            # 401/403 split is the only signal that separates "not
            # authenticating" from "authenticating with no entitlement".
            raise BOTFXError(
                "BOT refused the request (403). The token is being sent but the "
                "app is not approved for this API product.",
                status=403,
            )
        if response.status_code != 200:
            raise BOTFXError(
                f"BOT returned {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise BOTFXError(f"BOT response was not JSON: {exc}") from exc

    def fetch_daily_avg(
        self,
        start: date,
        end: date,
        currency: Optional[str] = "USD",
        throttle: float = THROTTLE_SECONDS,
        progress=None,
    ) -> RateTable:
        """Daily average rates over a range, in as many windows as it takes.

        `currency=None` asks for every currency the BOT publishes (19 of them)
        at no extra cost in requests.
        """
        if start < SERIES_START:
            start = SERIES_START
        rates: RateTable = {}
        windows = month_windows(start, end)
        self.check_budget(len(windows))
        for index, (window_start, window_end) in enumerate(windows):
            params = {
                "start_period": window_start.isoformat(),
                "end_period": window_end.isoformat(),
            }
            if currency:
                params["currency"] = currency
            payload = self._get(DAILY_AVG_PATH, params)
            detail = (
                payload.get("result", {}).get("data", {}).get("data_detail", [])
            )
            rates.update(parse_detail(detail))
            # Only every twelfth window, plus the last: a per-request line is
            # 290 of them, which is fine over \r on a terminal and a single
            # unreadable smear in a log file or a captured run.
            if progress and (index % 12 == 0 or index + 1 == len(windows)):
                progress(index + 1, len(windows), window_start, window_end, len(rates))
            if throttle and index + 1 < len(windows):
                time.sleep(throttle)
        return rates

    def verify(self) -> Dict[str, object]:
        """A one-call health check: is the token live, and what does it serve."""
        end = date.today()
        rates = self.fetch_daily_avg(end - timedelta(days=7), end, throttle=0)
        if not rates:
            return {"ok": False, "reason": "authenticated, but no rows returned"}
        latest = max(rates)
        return {
            "ok": True,
            "latest": latest,
            "thb_per_usd": baht_per(rates, latest, "USD"),
            "days": len(rates),
        }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Bank of Thailand FX rates")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check the token and report the latest day")
    rates_cmd = sub.add_parser("rates", help="print daily average rates")
    rates_cmd.add_argument("--currency", default="USD")
    rates_cmd.add_argument("--start", required=True)
    rates_cmd.add_argument("--end", required=True)
    args = parser.parse_args()

    provider = BOTFXProvider()
    try:
        if args.command == "verify":
            print(json.dumps(provider.verify(), indent=2))
            return 0
        rates = provider.fetch_daily_avg(
            datetime.strptime(args.start, "%Y-%m-%d").date(),
            datetime.strptime(args.end, "%Y-%m-%d").date(),
            currency=args.currency,
        )
        for day in sorted(rates):
            print(f"{day}  {rates[day]}")
        print(f"{len(rates)} day(s)")
        return 0
    except BOTFXError as exc:
        print(f"BOT: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
