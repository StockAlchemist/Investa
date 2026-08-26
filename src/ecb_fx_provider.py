# -*- coding: utf-8 -*-
"""European Central Bank euro reference rates — official daily FX (plan Phase 5.1).

Every FX rate the archive holds comes from Yahoo, which makes the currency
conversion behind every portfolio figure share a single point of failure with
the price feed. It has already failed once: `EUR=X`, `GBP=X` and `CNY=X` sat
frozen at their 15 Jun 2026 values until the Tier A backfill noticed in late
August, and a frozen rate is not an outage anyone sees — the numbers keep
rendering, quietly two months stale.

The ECB publishes its reference rates as two static files, no key and no
scraping:

    eurofxref-hist.zip      a CSV of every business day since 1999-01-04
    eurofxref-hist-90d.xml  the last 90 days, ~70 KB — what a nightly run wants

Both are EUR-based: the number against `USD` is dollars per euro. The archive
stores Yahoo-style USD-based pairs, so every rate here is a cross computed
through the euro (see `pair_rate`).

**Three limits worth knowing before trusting a number from here.**

*Coverage starts late for Asia.* THB and CNY were only added to the reference
list on 2005-04-01. The ledger opens in Jun 2002, so the ECB cannot re-price the
first three years of a Thai holding; only the Bank of Thailand goes back that
far, and its API needs a registered key.

*It is a 14:15 CET fix, not a close.* Measured against the archive's stored
Yahoo rates over ~5,500 overlapping days, the median disagreement is 0.21% and
the 95th percentile 0.8% — the timing difference, not an error in either. That
is why the ingester fills gaps rather than overwriting: rewriting a stored day
would move every historical portfolio figure by a fifth of a percent for no gain
in accuracy.

*It publishes on TARGET days.* European holidays have no rate even when New York
and Bangkok are trading, so a series from here has holes a market calendar does
not explain.

    python src/ecb_fx_provider.py rates --pair THB=X --days 5
"""

import csv
import io
import logging
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

ECB_HIST_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ECB_90D_XML_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml"
ECB_DAILY_XML_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

DEFAULT_TIMEOUT = 30

# The reference rates are quoted per euro, so the euro is its own unit and never
# appears as a column in the feed.
_EUR = "EUR"

_XML_NS = {"ecb": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}

SOURCE = "ecb"

# EUR-based rows: {'yyyy-MM-dd': {'USD': 1.1662, 'THB': 38.176, ...}}
RateTable = Dict[str, Dict[str, float]]


class ECBFXError(Exception):
    """The feed was unreachable or did not parse."""


def split_pair(pair: str) -> Optional[Tuple[str, str]]:
    """Yahoo-style pair name -> (base, quote), or None if it is not one.

    The archive uses two spellings for the same number, and both have to keep
    meaning what `portfolio_history` already assumes they mean:

        THB=X      THB per USD     (three letters: USD is the implied base)
        USDTHB=X   THB per USD     (the same series, spelled out)
        THBUSD=X   USD per THB     (the inverse)

    So a three-letter name is read as `USD{CUR}`, which is what makes `USD=X`
    resolve to a flat 1.0 rather than to whatever a provider happens to return
    for a currency against itself.
    """
    if not pair or not pair.upper().endswith("=X"):
        return None
    code = pair.upper()[:-2]
    if len(code) == 3 and code.isalpha():
        return "USD", code
    if len(code) == 6 and code.isalpha():
        return code[:3], code[3:]
    return None


def pair_rate(row: Dict[str, float], pair: str) -> Optional[float]:
    """The pair's rate for one day's EUR-based row, or None if uncovered.

    Both legs are crossed through the euro, so a pair is available only on a day
    the ECB published *both* currencies — which is why THB and CNY series here
    begin in Apr 2005 while EUR and JPY reach back to 1999.
    """
    legs = split_pair(pair)
    if not legs:
        return None
    base, quote = legs

    def leg(code: str) -> Optional[float]:
        if code == _EUR:
            return 1.0
        value = row.get(code)
        return value if value and value > 0 else None

    base_rate, quote_rate = leg(base), leg(quote)
    if base_rate is None or quote_rate is None:
        return None
    return quote_rate / base_rate


def pair_series(rates: RateTable, pair: str) -> List[Tuple[str, float]]:
    """(date, rate) for every day the pair is derivable, oldest first."""
    out = [
        (day, value)
        for day, row in rates.items()
        if (value := pair_rate(row, pair)) is not None
    ]
    out.sort()
    return out


def supported_pairs(rates: RateTable, candidates: Iterable[str]) -> List[str]:
    """Those candidates the feed can actually price on at least one day."""
    return [p for p in candidates if any(pair_rate(row, p) for row in rates.values())]


def parse_hist_csv(data: bytes) -> RateTable:
    """The historical CSV: one header of currency codes, one row per day."""
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = [cell.strip() for cell in next(reader)]
    except StopIteration:
        raise ECBFXError("ECB history CSV was empty")
    if not header or header[0].lower() != "date":
        raise ECBFXError(f"unexpected ECB CSV header: {header[:4]}")

    codes = header[1:]
    rates: RateTable = {}
    for row in reader:
        if not row or not row[0].strip():
            continue
        day = row[0].strip()
        values: Dict[str, float] = {}
        for code, cell in zip(codes, row[1:]):
            # 'N/A' marks a currency the ECB had not yet added, or has dropped.
            cell = (cell or "").strip()
            if not code or not cell or cell.upper() == "N/A":
                continue
            try:
                values[code] = float(cell)
            except ValueError:
                continue
        if values:
            rates[day] = values
    if not rates:
        raise ECBFXError("ECB history CSV held no rates")
    return rates


def parse_xml(data: bytes) -> RateTable:
    """The daily / 90-day XML: nested <Cube> elements, one level per day."""
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ECBFXError(f"ECB XML did not parse: {exc}") from exc

    rates: RateTable = {}
    for day_cube in root.iter(f"{{{_XML_NS['ecb']}}}Cube"):
        day = day_cube.get("time")
        if not day:
            continue
        values: Dict[str, float] = {}
        for cube in day_cube:
            code, rate = cube.get("currency"), cube.get("rate")
            if not code or not rate:
                continue
            try:
                values[code] = float(rate)
            except ValueError:
                continue
        if values:
            rates[day] = values
    if not rates:
        raise ECBFXError("ECB XML held no rates")
    return rates


class ECBFXProvider:
    """Reads the ECB's published reference-rate files. No key, no account."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, session=None):
        self.timeout = timeout
        self._session = session

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "Investa/1.0 (archive)"})
        return self._session

    def _get(self, url: str) -> bytes:
        try:
            response = self._get_session().get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ECBFXError(f"ECB request failed ({url}): {exc}") from exc
        return response.content

    def fetch_history(self) -> RateTable:
        """Every business day since 1999-01-04. ~640 KB zipped."""
        payload = self._get(ECB_HIST_ZIP_URL)
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise ECBFXError(f"no CSV inside {ECB_HIST_ZIP_URL}")
                data = archive.read(names[0])
        except zipfile.BadZipFile as exc:
            raise ECBFXError(f"ECB history was not a zip: {exc}") from exc
        return parse_hist_csv(data)

    def fetch_recent(self, window: str = "90d") -> RateTable:
        """The last 90 days (default) or just the latest published day."""
        url = ECB_90D_XML_URL if window == "90d" else ECB_DAILY_XML_URL
        return parse_xml(self._get(url))

    def verify(self) -> Dict[str, object]:
        """A one-call health check: is the feed up, and how current is it."""
        rates = self.fetch_recent("daily")
        day = max(rates)
        return {
            "ok": True,
            "latest": day,
            "currencies": len(rates[day]),
            "usd_per_eur": rates[day].get("USD"),
        }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ECB reference rates")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check the feed and report the latest day")
    rates_cmd = sub.add_parser("rates", help="print a pair's recent rates")
    rates_cmd.add_argument("--pair", default="THB=X")
    rates_cmd.add_argument("--days", type=int, default=10)
    rates_cmd.add_argument(
        "--history", action="store_true", help="use the full CSV, not the 90-day XML"
    )
    args = parser.parse_args()

    provider = ECBFXProvider()
    try:
        if args.command == "verify":
            print(json.dumps(provider.verify(), indent=2))
            return 0
        rates = provider.fetch_history() if args.history else provider.fetch_recent()
        series = pair_series(rates, args.pair)
        if not series:
            print(f"{args.pair}: not derivable from the ECB feed")
            return 1
        for day, rate in series[-args.days :]:
            print(f"{day}  {rate:.6f}")
        print(f"{len(series)} days, {series[0][0]} -> {series[-1][0]}")
        return 0
    except ECBFXError as exc:
        print(f"ECB: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
