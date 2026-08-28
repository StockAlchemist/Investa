# -*- coding: utf-8 -*-
"""Keeping the Thai fund NAV store current.

`scripts/backfill_fund_nav.py` filled `fund_nav` once, from 2000 to the day it
was run. Nothing kept it moving afterwards, so the series aged out from under
the valuation that reads it — four funds stalled 3-6 days behind while the
holdings table quietly fell back to a hand-entered scalar.

This module owns the two things both the backfill and the background worker
need to agree on:

  * **which local code is which SEC fund** — an alias or a sub-policy decided
    differently in two places would write one fund's history under another's
    name, which is unrecoverable without knowing it happened;
  * **how to top up** — refetch a short trailing window rather than only what
    is missing, because the SEC revises recently published NAVs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import config
from sec_thailand_provider import (
    FundMatch,
    SECThailandError,
    SECThailandNotConfiguredError,
    SECThailandProvider,
)

logger = logging.getLogger(__name__)

# The SEC's own NAV history does not usefully predate this, and the ledger's
# first transaction is 2002 — matching the archive's universe floor.
DEFAULT_START = date(2000, 1, 1)

# How far back a top-up refetches. The SEC restates NAVs published in the last
# few days, so resuming from exactly the last stored date would keep whichever
# provisional value happened to be there first.
TOP_UP_LOOKBACK_DAYS = 10

# Local fund codes that do not match the SEC's own abbreviation.
#
# Everything else resolves automatically, including share classes: SCBCHA-SSF is
# a *class* of project SCBCHAFUND and the resolver finds it by searching the stem
# and matching the class name exactly. Only a genuine naming difference belongs
# here, and each entry is a claim that two names are the same fund — get one
# wrong and a whole history of somebody else's NAVs lands under your code, which
# is why these are declared explicitly rather than guessed by fuzzy matching.
FUND_CODE_ALIASES = {
    # Local 'SCBRCTECH' vs the SEC's 'SCBRMCTECH' (SCB China Technology RMF,
    # M0295_2564) — a dropped 'M' against the sibling codes SCBRM1 and
    # SCBRMS&P500. NOT the same fund as SCBCTECH-SSF, which is the SSF class of
    # the non-RMF SCB China Technology.
    "SCBRCTECH": "SCBRMCTECH",
}

# Codes that are provident-fund sub-policies, NOT retail mutual funds.
#
# These are investment options inside the "Eastspring M Choice" pooled provident
# fund (SEC proj_id V0006_2552) — the platform whose logo appears on the
# statements. They are a genuine trap: several share a name with a retail
# Eastspring fund, so a lookup resolves happily and backfills the wrong
# instrument's entire history. ES-GQG did exactly that here — 2,694 rows of the
# retail Eastspring Global Quality Growth Fund were written under it before the
# trade prices settled it: across 25 monthly contributions not one matched the
# retail NAV, every one sat 3.6-6.4% below it, and sweeping the match +/- 5 days
# never brought the gap near zero. A settlement lag swings both ways; a
# one-sided gap that size is a different instrument.
#
# They are skipped rather than fetched because the SEC's PVD NAV endpoint
# (/v1/pvd/factsheet/{proj_id}/nav/{yyyyMMdd}) publishes month-end values only,
# and only from roughly 2024-12 to 2026-05 — about eighteen points against
# fifteen years of monthly contributions. Not enough to value a position with,
# and worse than nothing if it were mistaken for a real series.
#
# What prices them instead: nothing here. Their override entry carries metadata
# and no `price`, so the engine falls through to the ledger's own transaction
# prices — which for a monthly contribution plan *are* the sub-policy's NAV on
# each contribution date, carried forward between them.
PVD_SUB_POLICIES = {
    "ES-GQG": ("V0006_2552", "000625520014", "ตราสารทุน (equity)"),
    "ES-FIXED_INCOME": ("V0006_2552", "000625520029", "ตราสารหนี้ (fixed income)"),
    "ES-TRESURY": ("V0006_2552", "000625520036", "ตราสารหนี้ - ตลาดเงิน (money market)"),
    # ES-SET50 and ES-JUMBO25 are almost certainly M Choice policies too, but
    # both were sold before the PVD NAV window opens, so nothing here can
    # confirm which sub-policy each is. Left unlisted rather than guessed.
}


def discover_fund_codes() -> Dict[str, List[str]]:
    """{fund_code: [users who hold it]} from every user's manual overrides.

    Read from the override files rather than a hardcoded list so a fund added
    there is picked up without editing code. An entry that carries metadata
    only — no `price` — still counts: it names a held instrument, which is what
    this is looking for.
    """
    users_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR)
    found: Dict[str, List[str]] = {}
    if not os.path.isdir(users_dir):
        return found

    for username in sorted(os.listdir(users_dir)):
        path = os.path.join(
            users_dir, username, config.CONFIG_DIR, "manual_overrides.json"
        )
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                overrides = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(overrides, dict):
            continue
        for code in overrides.get("manual_price_overrides") or {}:
            found.setdefault(code, []).append(username)
    return found


def resolve_code(
    provider: SECThailandProvider, code: str
) -> Tuple[Optional[FundMatch], str]:
    """Resolve one local code. Returns (match_or_None, reason).

    `reason` is 'ok', 'pvd' (deliberately skipped), or the resolver's own
    unresolved reason — the caller decides whether that is worth reporting.
    A PVD skip is a decision, not a failure.
    """
    if code.upper() in PVD_SUB_POLICIES:
        return None, "pvd"

    lookup_code = FUND_CODE_ALIASES.get(code.upper(), code)
    match = provider.resolve_fund(lookup_code)
    return (match, "ok") if match.resolved else (None, match.matched_on)


def top_up(
    codes: Optional[List[str]] = None,
    lookback_days: int = TOP_UP_LOOKBACK_DAYS,
    today: Optional[date] = None,
    db: Optional[Any] = None,
    provider: Optional[SECThailandProvider] = None,
) -> Dict[str, int]:
    """Fetch NAVs published since each fund's stored series ends.

    Returns {fund_code: rows_written} for the funds actually touched. A fund
    with nothing stored yet is left alone: filling twenty-five years from a
    background worker is the backfill script's job, and doing it here would
    make a routine tick take minutes.

    Every failure is contained per fund. One unreachable fund must not stop the
    other four from catching up, and none of them may take down the caller.
    """
    from market_db import MarketDatabase

    db = db or MarketDatabase()
    coverage = db.get_fund_nav_coverage()
    if not coverage:
        logger.debug("Fund NAV top-up: nothing stored yet, leaving it to the backfill")
        return {}

    wanted = [c for c in (codes or sorted(coverage)) if c in coverage]
    if not wanted:
        return {}

    provider = provider or SECThailandProvider()
    end = today or date.today()
    written_by_code: Dict[str, int] = {}

    for code in wanted:
        _first, last, _count = coverage[code]
        try:
            start = date.fromisoformat(str(last)) - timedelta(days=lookback_days)
        except ValueError:
            logger.warning(
                f"Fund NAV top-up: unreadable last date for {code}: {last!r}"
            )
            continue
        if start > end:
            continue

        try:
            match, reason = resolve_code(provider, code)
        except SECThailandNotConfiguredError:
            # No key configured — nothing here can work, and saying so once per
            # cycle for five funds is noise.
            logger.debug("Fund NAV top-up: SEC_TH_API_KEY not configured")
            return written_by_code
        except SECThailandError as exc:
            logger.warning(f"Fund NAV top-up: lookup failed for {code}: {exc}")
            continue

        if match is None:
            if reason != "pvd":
                logger.warning(f"Fund NAV top-up: {code} did not resolve ({reason})")
            continue

        try:
            rows = provider.fetch_nav(
                match.proj_id, start, end, fund_class_name=match.fund_class_name
            )
        except SECThailandError as exc:
            logger.warning(f"Fund NAV top-up: fetch failed for {code}: {exc}")
            continue

        if not rows:
            continue

        try:
            written = db.upsert_fund_nav(
                code, [(r["date"], r["nav"]) for r in rows if r.get("nav") is not None]
            )
        except Exception as exc:
            logger.warning(f"Fund NAV top-up: write failed for {code}: {exc}")
            continue

        if written:
            written_by_code[code] = written
            logger.info(
                f"Fund NAV top-up: {code} +{written} rows "
                f"({rows[0]['date']} .. {rows[-1]['date']})"
            )

    return written_by_code
