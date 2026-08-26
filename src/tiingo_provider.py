# -*- coding: utf-8 -*-
"""Tiingo — the second opinion that closes the manual repair loop (plan 5.3).

The archive's split repair has been manual since D1 was re-diagnosed, and
deliberately so: price data alone cannot say which of two disagreeing bases is
right (the one time it was automated from prices, 19 of 24 symbols got worse),
and the only reference that could adjudicate was IBKR, which retail accounts
cannot automate. `check_split_consistency.py` therefore reports and waits — 65
findings currently sit in that queue.

Tiingo can answer them. Its daily endpoint returns, per bar:

    close        the price as actually traded, NOT back-adjusted
    adjClose     split- and dividend-adjusted
    splitFactor  the ratio, on the ex-date itself
    divCash      cash dividend per share

which is the raw-plus-actions model Phase 1 rebuilt the archive around, from a
provider rather than reconstructed. Verified against two splits the archive
already knows: AAPL closes 500.04 on 2020-08-27 with `splitFactor` 4.0 on 08-31,
and NVDA 1208.88 on 2024-06-07 with 10.0 on 06-10 — both matching what
`get_ohlcv(adjust='none')` reproduces locally.

**The basis has to be converted before it can adjudicate, and this is the whole
trap.** `repair_bars_against_reference.py` acts when a stored bar differs from
the reference by exactly one of the symbol's own split ratios. The archive holds
Yahoo's split-adjusted prices; Tiingo's `close` is raw. Hand it the raw series
and *every* pre-split bar differs by exactly the split ratio — so the tool would
"repair" the entire history onto the raw basis. That is not a repair, it is a
migration, and it is the specific thing that script's docstring refuses to do.
So `split_adjusted` below re-applies the same arithmetic the archive uses
(`raw(d) ÷ Π{ratio : ex_date > d}`) and the reference is stored on the archive's
own basis.

`adjClose` is not a substitute: it folds in dividends too, so it would disagree
with the archive by a factor that is not any split ratio, and every bar would be
silently ignored instead of adjudicated.

Coverage is US-only — `PTT` is a 404 — so this settles US findings and says
nothing about SET. Free tier: `TIINGO_API_KEY=…` in `.env`.

    python src/tiingo_provider.py verify
    python src/tiingo_provider.py prices --symbol AAPL --start 2020-08-27 --end 2020-09-02
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

import requests

from config import TIINGO_API_KEY

logger = logging.getLogger(__name__)

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"
DEFAULT_TIMEOUT = 30

SOURCE = "tiingo"

# Tiingo's series start well before anything the archive needs to adjudicate.
DEFAULT_START = date(1990, 1, 1)


class TiingoError(Exception):
    """The API refused the request or answered something unreadable."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class TiingoNotConfiguredError(TiingoError):
    """No token. A setup problem, not an outage."""


class TiingoSymbolUnknown(TiingoError):
    """Tiingo does not carry this ticker — a 404, and expected for non-US."""


def split_adjusted(rows: List[dict]) -> Dict[str, float]:
    """Raw closes -> the archive's basis: split-adjusted, dividends untouched.

    `raw(d) ÷ Π{ratio : ex_date > d}`, the same arithmetic `market_db` applies
    when serving `adjust='split'`. Walking the series backwards makes that
    product a running one: every bar strictly before an ex-date carries it.

    Requires the *whole* series, not a window around the date in question — a
    split after the window would be missed and the reference would sit on a
    basis of its own, which is worse than having no reference at all.
    """
    out: Dict[str, float] = {}
    factor = 1.0
    for row in sorted(rows, key=lambda r: r["date"], reverse=True):
        day = str(row.get("date", ""))[:10]
        close = row.get("close")
        if not day or close is None:
            continue
        out[day] = float(close) / factor
        # The ratio applies to every bar *before* its ex-date, so it joins the
        # running product only after this bar has been priced.
        ratio = row.get("splitFactor")
        try:
            ratio = float(ratio) if ratio is not None else 1.0
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio and ratio != 1.0:
            factor *= ratio
    return out


class TiingoProvider:
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key if api_key is not None else TIINGO_API_KEY
        self.timeout = timeout
        self._session = None
        self._calls = 0

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def calls_made(self) -> int:
        """Requests this process has made — the free tier meters them."""
        return self._calls

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self.api_key}",
                }
            )
        return self._session

    def _get(self, path: str, params: Dict[str, object]) -> object:
        if not self.is_configured():
            raise TiingoNotConfiguredError(
                "No TIINGO_API_KEY. Register at https://www.tiingo.com and put the "
                "token in .env."
            )
        try:
            self._calls += 1
            response = self._get_session().get(
                f"{TIINGO_BASE}{path}", params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise TiingoError(f"Tiingo request failed: {exc}") from exc

        if response.status_code == 404:
            raise TiingoSymbolUnknown(
                f"Tiingo does not carry this ticker ({path})", status=404
            )
        if response.status_code == 429:
            raise TiingoError(
                "Tiingo rate limit reached. The free tier meters requests per hour "
                "and unique symbols per month; re-run later — the reference load is "
                "idempotent, so nothing is duplicated.",
                status=429,
            )
        if response.status_code != 200:
            raise TiingoError(
                f"Tiingo returned {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise TiingoError(f"Tiingo response was not JSON: {exc}") from exc

    def fetch_prices(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[dict]:
        """Daily bars, raw close plus `splitFactor`/`divCash`, oldest first."""
        params: Dict[str, object] = {
            "startDate": (start or DEFAULT_START).isoformat()
        }
        if end:
            params["endDate"] = end.isoformat()
        payload = self._get(f"/{symbol}/prices", params)
        if not isinstance(payload, list):
            raise TiingoError(f"Unexpected payload for {symbol}: {type(payload)}")
        return payload

    def reference_closes(self, symbol: str) -> Dict[str, float]:
        """The symbol's whole history on the archive's basis, ready to compare."""
        return split_adjusted(self.fetch_prices(symbol))

    def verify(self) -> Dict[str, object]:
        """One call: is the token live, and does a known split come back right."""
        rows = self.fetch_prices(
            "AAPL", date(2020, 8, 27), date(2020, 9, 2)
        )
        by_day = {str(r["date"])[:10]: r for r in rows}
        pre = by_day.get("2020-08-27", {})
        return {
            "ok": bool(rows),
            "bars": len(rows),
            "aapl_2020_08_27_raw_close": pre.get("close"),
            "split_on_2020_08_31": by_day.get("2020-08-31", {}).get("splitFactor"),
        }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Tiingo daily prices")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check the token against a known split")
    p = sub.add_parser("prices", help="print raw and split-adjusted closes")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = parser.parse_args()

    def as_date(value):
        return datetime.strptime(value, "%Y-%m-%d").date() if value else None

    provider = TiingoProvider()
    try:
        if args.command == "verify":
            print(json.dumps(provider.verify(), indent=2))
            return 0
        rows = provider.fetch_prices(
            args.symbol, as_date(args.start), as_date(args.end)
        )
        adjusted = split_adjusted(rows)
        print(f"{'date':12} {'raw':>12} {'split-adj':>12} {'split':>7}")
        for row in rows:
            day = str(row["date"])[:10]
            print(
                f"{day:12} {row['close']:12.4f} {adjusted.get(day, float('nan')):12.4f} "
                f"{row.get('splitFactor', 1.0):7}"
            )
        return 0
    except TiingoError as exc:
        print(f"Tiingo: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
