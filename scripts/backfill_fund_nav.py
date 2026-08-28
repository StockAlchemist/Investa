#!/usr/bin/env python3
"""Backfill Thai mutual fund NAVs from the SEC Open API (plan Phase 3).

Five held funds are valued from a single hand-entered number in each user's
`manual_overrides.json`. There is no series behind them, so every historical
valuation of those positions is flat at today's NAV: allocation history,
drawdown and TWR are all wrong for whatever the position was worth, across a
ledger that starts in 2002.

Nobody sells this data. Yahoo, Stooq and Tiingo have no Thai SSF/RMF NAVs at
all, which is why this ranks ahead of widening the US price archive: it is the
only holding class with no provider, and NAVs not captured cannot be
reconstructed later.

Four of the five are retail mutual funds and backfill from the SEC's daily NAV
API. The fifth, ES-GQG, turned out to be a provident-fund sub-policy wearing a
retail fund's name — see PVD_SUB_POLICIES below for why that distinction is
worth this much comment.

Fund codes are read from the users' manual_overrides.json rather than hardcoded,
so a fund added there is picked up without editing this script. Each code is
resolved to a SEC proj_id by exact abbreviation match (share classes included);
anything ambiguous is reported and skipped rather than guessed at.

    python scripts/backfill_fund_nav.py --dry-run
    python scripts/backfill_fund_nav.py --apply
    python scripts/backfill_fund_nav.py --apply --fund SCBRM1 --start 2015-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Fund resolution policy — the alias table, the provident-fund skip list, the
# history floor and the override-file scan — lives in `src/fund_nav_sync.py`,
# shared with the background top-up worker. Two copies of an alias is how one
# fund's history ends up filed under another's name.
from fund_nav_sync import (  # noqa: E402
    DEFAULT_START,
    FUND_CODE_ALIASES,
    PVD_SUB_POLICIES,
    discover_fund_codes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply", action="store_true", help="write (default is a dry run)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report only — the default, accepted so the flag can be explicit",
    )
    parser.add_argument(
        "--fund", action="append", help="limit to this code (repeatable)"
    )
    parser.add_argument("--start", help="YYYY-MM-DD (default 2000-01-01)")
    parser.add_argument("--end", help="YYYY-MM-DD (default today)")
    args = parser.parse_args()
    if args.dry_run:
        args.apply = False

    from market_db import MarketDatabase
    from sec_thailand_provider import (
        SECThailandError,
        SECThailandNotConfiguredError,
        SECThailandProvider,
    )

    provider = SECThailandProvider()
    if not provider.is_configured:
        print(
            "SEC_TH_API_KEY is not set in .env.\n"
            "Register for a free subscription key at https://secopendata.sec.or.th/sec-open-apis "
            "(self-service, no review), subscribe to the Fund Daily Info API, then:\n"
            "    SEC_TH_API_KEY=<your key>"
        )
        return 2

    codes = args.fund or sorted(discover_fund_codes())
    if not codes:
        print(
            "No manually-priced fund codes found in any user's manual_overrides.json."
        )
        return 0

    start = (
        datetime.strptime(args.start, "%Y-%m-%d").date()
        if args.start
        else DEFAULT_START
    )
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    db = MarketDatabase()
    existing = db.get_fund_nav_coverage()

    print(f"{len(codes)} fund code(s); NAV window {start} .. {end}\n")

    resolved: Dict[str, Any] = {}
    for code in codes:
        pvd = PVD_SUB_POLICIES.get(code.upper())
        if pvd:
            proj, sub, category = pvd
            print(
                f"  {code:16} SKIPPED — provident-fund sub-policy "
                f"{proj}/{sub} ({category});"
            )
            print(
                f"  {'':16}    the SEC publishes month-end NAVs only, ~2024-12 onward"
            )
            resolved[code] = None
            continue

        lookup_code = FUND_CODE_ALIASES.get(code.upper(), code)
        try:
            match = provider.resolve_fund(lookup_code)
        except SECThailandNotConfiguredError as exc:
            print(exc)
            return 2
        except SECThailandError as exc:
            print(f"  {code:16} lookup failed: {exc}")
            resolved[code] = None
            continue

        resolved[code] = match if match.resolved else None
        have = existing.get(code)
        have_desc = (
            f"have {have[2]} rows {have[0]}..{have[1]}" if have else "no rows yet"
        )
        alias_note = f" (as {lookup_code})" if lookup_code != code else ""

        if match.resolved:
            name = (match.profile or {}).get("proj_name_en") or ""
            klass = f" class={match.fund_class_name}" if match.fund_class_name else ""
            print(
                f"  {code:16} -> {match.proj_id:14}{klass}{alias_note}  ({have_desc})"
            )
            print(f"  {'':16}    {name[:60]}  [matched on {match.matched_on}]")
        else:
            print(f"  {code:16} -> UNRESOLVED{alias_note}  [{match.matched_on}]")
            if match.candidates:
                print(f"  {'':16}    candidates: {', '.join(match.candidates[:10])}")

    # A PVD skip is a decision, not a failure — don't send the reader off to
    # add an alias for something deliberately excluded.
    unresolved = [
        c for c, m in resolved.items() if not m and c.upper() not in PVD_SUB_POLICIES
    ]
    if unresolved:
        print(
            f"\n{len(unresolved)} code(s) did not resolve to a single fund: "
            f"{', '.join(unresolved)}\n"
            "Run `python src/sec_thailand_provider.py lookup <stem>` to see the "
            "candidates, then add an entry to FUND_CODE_ALIASES in this script."
        )

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to backfill.")
        return 0

    print("\nFetching NAVs:")
    total_written = 0
    failures: List[str] = []
    for code, match in resolved.items():
        if not match:
            continue
        print(f"  {code:16} ", end="", flush=True)
        try:
            rows = provider.fetch_nav(
                match.proj_id,
                start,
                end,
                fund_class_name=match.fund_class_name,
            )
        except SECThailandError as exc:
            print(f"FAILED: {exc}")
            failures.append(code)
            continue

        if not rows:
            print("no NAVs returned for this window")
            continue

        written = db.upsert_fund_nav(code, [(r["date"], r["nav"]) for r in rows])
        total_written += written
        print(f"{written} NAVs, {rows[0]['date']} .. {rows[-1]['date']}")

    print(f"\nWrote {total_written} NAV rows.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        return 1

    coverage = db.get_fund_nav_coverage()
    print("\nCoverage now:")
    for code in sorted(coverage):
        first, last, count = coverage[code]
        print(f"  {code:16} {count:6} rows  {first} .. {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
