# -*- coding: utf-8 -*-
"""Thai SEC Open API client — daily NAVs for Thai mutual funds.

Investa holds five Thai SSF/RMF funds (SCBRMS&P500, SCBRM1, SCBRCTECH,
SCBCHA-SSF, ES-GQG) that are priced from a single hand-entered number in
`manual_overrides.json`. No commercial provider carries Thai fund NAVs — not
Yahoo, not Stooq, not Tiingo — so every historical valuation of those positions
is flat-lined at today's price, which is quietly wrong for TWR, drawdown and
allocation history across a twenty-four year ledger.

The SEC publishes them itself. This module reads two v2 endpoints:

    GET /v2/fund/general-info/profiles   — resolve a fund abbreviation to proj_id
    GET /v2/fund/daily-info/nav          — daily NAV over a date range

`last_val` is the NAV per unit, which is the number that corresponds to a share
price; `net_asset` is the fund's total size and is not what a holding is valued
at. Both are kept, only the first is used as the price.

Credentials: a free, self-service subscription key from
https://secopendata.sec.or.th/sec-open-apis — no application review, unlike a
broker API.

    SEC_TH_API_KEY=...        # in .env

    python src/sec_thailand_provider.py lookup SCBRM1
    python src/sec_thailand_provider.py nav M0004_2559 --start 2020-01-01

Two migrations to keep straight, because most write-ups online still describe
the retired setup:

  * The developer portal moved. `api-portal.sec.or.th` was discontinued on
    30 Jun 2026 and no longer resolves in DNS at all — registration is now at
    secopendata.sec.or.th. The API gateway (`api.sec.or.th`) did not move.
  * The v1 NAV endpoint (`/FundDailyInfo/{proj_id}/dailynav/{date}`) took one
    call per fund per day — roughly 30,000 calls to backfill five funds over
    twenty-five years. v2 takes a date range and cursor-paginates, which is the
    same job in a few dozen calls.

If the portal is unreachable, the SEC's contact for API access is
repcenter@sec.or.th.
"""

import logging
import time
from datetime import date, datetime
from typing import Any, Dict, Iterator, List, NamedTuple, Optional

import requests

from config import SEC_TH_API_KEY

logger = logging.getLogger(__name__)


class FundMatch(NamedTuple):
    """
    Outcome of resolving a local fund code against the SEC catalogue.

    `fund_class_name` is not decoration: a project's share classes each have
    their own NAV, so fetching without the class a holding actually belongs to
    blends several series together.

    `matched_on` records how it resolved — 'abbr', 'class', or one of the
    unresolved reasons ('ambiguous-class', 'ambiguous-project', 'no-match') —
    so the caller can explain a failure instead of silently skipping.
    """

    proj_id: Optional[str]
    fund_class_name: Optional[str]
    matched_on: str
    profile: Optional[Dict[str, Any]]
    candidates: List[str]

    @property
    def resolved(self) -> bool:
        return bool(self.proj_id)

BASE_URL = "https://api.sec.or.th"
PROFILES_PATH = "/v2/fund/general-info/profiles"
NAV_PATH = "/v2/fund/daily-info/nav"

# The published limit is 3,000 calls per 300 seconds. A backfill is a background
# job with no user waiting on it, so it runs far below that rather than near it:
# tripping the limit mid-backfill costs more time than the pacing does.
MIN_SECONDS_BETWEEN_CALLS = 0.15

MAX_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30


class SECThailandError(Exception):
    """A SEC Open API call failed."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status = status

    @property
    def is_config_error(self) -> bool:
        """A missing or rejected key — retrying never helps."""
        return self.status in (401, 403)


class SECThailandNotConfiguredError(SECThailandError):
    """No subscription key. Raised before any network call."""


class SECThailandProvider:
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key or SEC_TH_API_KEY
        self.timeout = timeout
        self._session: Optional[requests.Session] = None
        self._last_call = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Ocp-Apim-Subscription-Key": self.api_key or "",
                    "Accept": "application/json",
                }
            )
            self._session = session
        return self._session

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_configured:
            raise SECThailandNotConfiguredError(
                "Thai SEC API is not configured. Set SEC_TH_API_KEY in .env — "
                "get a free key at https://secopendata.sec.or.th/sec-open-apis"
            )

        self._throttle()
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        try:
            response = self._get_session().get(
                f"{BASE_URL}{path}", params=clean, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise SECThailandError(f"Request failed: {exc}") from exc

        # 204 No Content is how this API says "your filter matched nothing" —
        # an empty result, not a failure. Treating it as an error turned an
        # ordinary "no such fund abbreviation" into a hard lookup failure.
        if response.status_code == 204 or not response.content:
            return {"items": [], "next_cursor": ""}

        if response.status_code != 200:
            body = response.text[:300]
            raise SECThailandError(
                f"HTTP {response.status_code} from {path}: {body}",
                status=response.status_code,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise SECThailandError(f"Non-JSON response from {path}") from exc

    def _paginate(self, path: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Yield every item across cursor pages.

        The API signals the end with an empty `next_cursor`. A page that comes
        back with items but no cursor change would loop forever, so the cursor
        is required to advance.
        """
        cursor: Optional[str] = None
        seen_cursors = set()

        while True:
            page_params = dict(params)
            page_params["page_size"] = min(
                int(params.get("page_size", MAX_PAGE_SIZE)), MAX_PAGE_SIZE
            )
            if cursor:
                page_params["next_cursor"] = cursor

            payload = self._get(path, page_params)
            items = payload.get("items") or []
            for item in items:
                yield item

            cursor = payload.get("next_cursor") or ""
            if not cursor or not items:
                return
            if cursor in seen_cursors:
                logger.warning(
                    "SEC API returned a repeated cursor; stopping pagination"
                )
                return
            seen_cursors.add(cursor)

    # --- fund lookup -------------------------------------------------------

    def lookup_fund(
        self, query: str, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Candidate funds for an abbreviation or name fragment.

        `project_info` is an exact match on proj_id and a partial match on the
        Thai/English names and abbreviation, so a fund code like 'SCBRM1' finds
        its project without knowing the id.
        """
        params: Dict[str, Any] = {"project_info": query, "page_size": MAX_PAGE_SIZE}
        if active_only:
            params["fund_status"] = "Registered"

        results: List[Dict[str, Any]] = []
        for item in self._paginate(PROFILES_PATH, params):
            results.append(
                {
                    "proj_id": item.get("proj_id"),
                    "abbr": item.get("proj_abbr_name"),
                    "name_en": item.get("proj_name_en"),
                    "name_th": item.get("proj_name_th"),
                    "status": item.get("fund_status"),
                    "amc": item.get("comp_name_en"),
                    "regis_date": item.get("regis_date"),
                    "cancel_date": item.get("cancel_date"),
                }
            )
        return results

    def _profile_rows(self, query: str) -> List[Dict[str, Any]]:
        """Raw profile rows — one per share class, so classes stay visible."""
        return list(
            self._paginate(
                PROFILES_PATH, {"project_info": query, "page_size": MAX_PAGE_SIZE}
            )
        )

    def resolve_fund(self, code: str) -> FundMatch:
        """
        Resolve a local fund code to a project and, where it matters, a class.

        Thai funds are a project with several share classes, and the classes are
        what an investor actually holds: SCBCHAFUND (one project) carries
        SCBCHA, SCBCHA-SSF, SCBCHAR, SCBCHAP and four more, each with its own
        NAV. A code like `SCBCHA-SSF` therefore names a *class*, not a project —
        `project_info` does not search class names, so a direct lookup finds
        nothing at all and pulling the project without a class filter would
        blend eight different NAV series into one.

        Resolution, in order:

          1. exact match on `proj_abbr_name` — the ordinary case;
          2. failing that, search the stem before the last '-' and look for an
             exact match on `fund_class_name`, which is what resolves
             `SCBCHA-SSF` -> project SCBCHAFUND, class SCBCHA-SSF.

        Anything else returns unresolved with the candidates attached. Guessing
        is the one outcome worth avoiding: a partial search for `SCBRM1` also
        returns `SCBRM10`, and `ES-GQG` sits beside `ES-GQG-UH` and `ES-GQGRMF`,
        so a near-miss silently backfills a different fund's entire history.
        """
        target = code.strip().upper()
        rows = self._profile_rows(code)

        exact_abbr = [
            r for r in rows if (r.get("proj_abbr_name") or "").strip().upper() == target
        ]
        if exact_abbr:
            proj_ids = {r.get("proj_id") for r in exact_abbr}
            classes = {
                (r.get("fund_class_name") or "").strip() for r in exact_abbr
            } - {""}
            if len(proj_ids) == 1:
                proj_id = exact_abbr[0]["proj_id"]
                # A single 'main' class needs no filter; several classes under an
                # abbreviation match is genuinely ambiguous and must not be
                # resolved by picking one.
                real_classes = {c for c in classes if c.lower() != "main"}
                if not real_classes:
                    return FundMatch(proj_id, None, "abbr", exact_abbr[0], [])
                if len(real_classes) == 1:
                    return FundMatch(
                        proj_id, real_classes.pop(), "abbr", exact_abbr[0], []
                    )
                return FundMatch(
                    None,
                    None,
                    "ambiguous-class",
                    exact_abbr[0],
                    sorted(real_classes),
                )
            logger.warning(f"{code}: abbreviation maps to {len(proj_ids)} projects")
            return FundMatch(None, None, "ambiguous-project", None, sorted(map(str, proj_ids)))

        # The code may name a share class. Search the stem before the last '-'
        # ('SCBCHA-SSF' -> 'SCBCHA') and match the class exactly.
        stem = target.rsplit("-", 1)[0] if "-" in target else target
        class_rows = rows if stem == target else self._profile_rows(stem)
        exact_class = [
            r
            for r in class_rows
            if (r.get("fund_class_name") or "").strip().upper() == target
        ]
        if exact_class:
            row = exact_class[0]
            return FundMatch(
                row.get("proj_id"),
                (row.get("fund_class_name") or "").strip(),
                "class",
                row,
                [],
            )

        seen = sorted(
            {
                str(r.get("proj_abbr_name"))
                for r in (rows or class_rows)
                if r.get("proj_abbr_name")
            }
        )
        return FundMatch(None, None, "no-match", None, seen)

    def resolve_proj_id(self, abbreviation: str) -> Optional[str]:
        """Back-compat shim: the project id only, ignoring any share class."""
        return self.resolve_fund(abbreviation).proj_id

    # --- NAVs --------------------------------------------------------------

    def fetch_nav(
        self,
        proj_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        fund_class_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Daily NAV rows for one fund over a date range.

        `nav` is `last_val`, the NAV per unit — the figure a holding is valued
        at. `net_asset` (total fund size) rides along for reference but must not
        be mistaken for a price.
        """
        params: Dict[str, Any] = {"proj_id": proj_id, "page_size": MAX_PAGE_SIZE}
        if start_date:
            params["start_nav_date"] = start_date.isoformat()
        if end_date:
            params["end_nav_date"] = end_date.isoformat()
        if fund_class_name:
            params["fund_class_name"] = fund_class_name

        rows: List[Dict[str, Any]] = []
        for item in self._paginate(NAV_PATH, params):
            nav_date = (item.get("nav_date") or "")[:10]
            nav = item.get("last_val")
            if not nav_date or nav is None:
                continue
            try:
                nav_value = float(nav)
            except (TypeError, ValueError):
                continue
            if nav_value <= 0:
                continue
            rows.append(
                {
                    "date": nav_date,
                    "nav": nav_value,
                    "net_asset": item.get("net_asset"),
                    "sell_price": item.get("sell_price"),
                    "buy_price": item.get("buy_price"),
                    "fund_class_name": item.get("fund_class_name"),
                }
            )

        rows.sort(key=lambda r: r["date"])
        return rows

    def verify(self) -> Dict[str, Any]:
        """Credential / connectivity check with an actionable failure."""
        result: Dict[str, Any] = {"configured": self.is_configured}
        if not self.is_configured:
            result["error"] = (
                "SEC_TH_API_KEY is not set in .env — free key at "
                "https://secopendata.sec.or.th/sec-open-apis"
            )
            return result
        try:
            payload = self._get(PROFILES_PATH, {"page_size": 1})
            result["ok"] = True
            result["sample_count"] = len(payload.get("items") or [])
        except SECThailandError as exc:
            result["ok"] = False
            result["error"] = str(exc)
            result["is_config_error"] = exc.is_config_error
        return result


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Thai SEC Open API client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="check the subscription key")

    look = sub.add_parser("lookup", help="find a fund's proj_id")
    look.add_argument("query")

    nav = sub.add_parser("nav", help="fetch daily NAVs")
    nav.add_argument("proj_id")
    nav.add_argument("--start")
    nav.add_argument("--end")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    provider = SECThailandProvider()

    def as_date(value: Optional[str]) -> Optional[date]:
        return datetime.strptime(value, "%Y-%m-%d").date() if value else None

    try:
        if args.cmd == "verify":
            output: Any = provider.verify()
        elif args.cmd == "lookup":
            output = provider.lookup_fund(args.query)
        else:
            output = provider.fetch_nav(
                args.proj_id, as_date(args.start), as_date(args.end)
            )
    except SECThailandNotConfiguredError as exc:
        print(exc)
        return 2
    except SECThailandError as exc:
        print(f"SEC API error: {exc}")
        return 1

    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
