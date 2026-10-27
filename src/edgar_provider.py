# -*- coding: utf-8 -*-
"""
SEC EDGAR XBRL fundamentals.

Why this exists: yfinance returns only 4–5 annual periods per company (measured:
AAPL 5, MSFT 4, JPM 4). Buffett-style quality assessment is about *durability* —
fifteen years of consistent returns on capital, not one good year — so a 4-year
window cannot support the ranking this feeds. EDGAR's XBRL company facts carry
~17 annual periods (2009 onward, when XBRL became mandatory) for free, with no
API key. FMP would have been the obvious alternative but the configured key
returns HTTP 402 on statement endpoints.

Two ingest paths:
  * `ingest_bulk` reads the SEC's ~1.4 GB `companyfacts.zip`, which contains one
    JSON per filer. This is the right path for a full build — 5,600 individual
    downloads would move ~30 GB and take hours.
  * `ingest_company` refreshes a single CIK from the REST API, for incremental
    top-ups once the bulk load exists.

Both funnel into the same SQLite store, keyed so a restatement never silently
overwrites the originally-filed number: every (cik, tag, period_end, accession)
is its own row. The resolver defaults to the most recently filed value, but the
originals remain available, which is what makes an honest point-in-time backtest
possible later (principle P8).

`get_statements` returns income/balance/cashflow frames using **yfinance's own
row labels and column ordering**, so `financial_ratios.calculate_key_ratios_timeseries`,
the DCF and the Graham model all consume EDGAR data with no changes.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import re

import pandas as pd

import config
import investa_core
from db_utils import get_db_connection
from edgar_concepts import (  # noqa: F401 -- read by the Rust port at call time
    BALANCE_CONCEPTS,
    BANK_CONCEPTS,
    CASHFLOW_CONCEPTS,
    INCOME_CONCEPTS,
    REIT_CONCEPTS,
    all_concepts,
    all_tags,
)
from edgar_http import get_user_agent, sec_get_json  # noqa: F401 -- read by the Rust port at call time

_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# The ranking is annual by design: quarterlies would let a company's fiscal
# calendar leak into cross-sectional comparisons. The quarterly store below is a
# separate table serving the per-stock statements view only, and nothing in the
# ranking reads it.
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})

# A "duration" fact is annual if it spans roughly a year. Filers are not exact:
# 52/53-week retail calendars produce 364- and 371-day years.
_MIN_ANNUAL_DAYS = 300
_MAX_ANNUAL_DAYS = 400

# --- quarterly ---------------------------------------------------------------
#
# Quarterly facts live in their own table, keyed on the period *start* as well
# as its end. A 10-Q reports the three-month and the year-to-date figure for the
# same tag, period end and accession, so one table keyed only on the end would
# have them overwrite each other — and merging quarters into the annual table
# would collide a fiscal Q4 with its own fiscal year, since both end on the same
# day.
_QUARTERLY_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})

# One reported quarter. 13 weeks is 91 days; 4-4-5 retail calendars and the
# 14-week quarter that keeps a 52/53-week year aligned widen the band.
#
# The ceiling is set by the 12/12/12/16-week calendar, not by the 14-week case: a
# filer on it closes a *16*-week fourth quarter, 112 days, and at 100 days that
# quarter was neither taken as filed nor derivable. Costco's Q4 was missing from
# every one of eighteen fiscal years. There is no risk of catching a rung of the
# year-to-date ladder instead, because the shortest of those is a half year.
_MIN_QUARTER_DAYS = 80
_MAX_QUARTER_DAYS = 120

# The year-to-date spans a 10-Q reports alongside (or instead of) the quarter:
# six months, nine months, and — from the 10-K — the full year.
_MAX_YTD_DAYS = _MAX_ANNUAL_DAYS

# Two period ends this close together are one period, tagged twice. NVIDIA's
# FY2012 10-K re-filed Q2 FY2011 under an end of 2010-07-31 where its own three
# earlier filings — and every balance-sheet instant for the quarter — say
# 2010-08-01. Keyed on the end, that one typo becomes a second $811m quarter, and
# the four columns behind a trailing-twelve-month figure then covered six months.
# Real quarter ends are never within a week of each other, so nothing a filer
# meant can be merged by this.
_PERIOD_END_TOLERANCE_DAYS = 3

# Recorded instead of a row count when the request itself failed, which a count
# of zero cannot distinguish from a filer that genuinely tags no quarterly XBRL.
# Only the first deserves a quick retry; the second has given its answer.
_QUARTERLY_FETCH_FAILED = -1

# Two quarter ends this far apart are consecutive. Anything wider means a gap in
# the year-to-date chain, and differencing across it would invent a number. The
# gap between consecutive ends *is* one quarter's length, so it shares the band
# above rather than keeping a second copy of it that can drift out of step —
# which is how the 16-week fourth quarter came to be rejected twice over.
_MIN_QUARTER_GAP_DAYS = _MIN_QUARTER_DAYS
_MAX_QUARTER_GAP_DAYS = _MAX_QUARTER_DAYS

# Tags reporting an average over their period rather than a sum across it. Every
# other duration fact here is a flow — revenue earned, cash moved — and a year is
# the sum of its quarters. A weighted-average share count is not: differencing it
# measures the drift between two averages, which is nothing at all.
_NON_ADDITIVE_TAGS = frozenset(
    {
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstandingBasic",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfSharesOutstanding",
    }
)

_DB_FILENAME = "edgar_facts.db"

# One filing's contribution to a tag: ((filed, accession), {span: (value, unit)}).
# The key matters because a split basis belongs to the *filing*, not to any one
# period in it — which is what lets a figure the share count never covered still
# be put on the right basis.
Filing = Tuple[Tuple[str, str], Dict[Tuple[str, str], Tuple[float, str]]]

# --- point-in-time view -----------------------------------------------------
#
# Every fact carries the date it was filed, so restricting reads to `filed <=
# some date` reconstructs what an investor could actually have known then. That
# is the whole basis of an honest backtest (P8): without it, a 2015 ranking
# would be scored on numbers restated in 2019 and on fiscal years not yet
# reported, and would look far cleverer than it was.
#
# A ContextVar rather than a module global: it defaults to None in every thread
# that did not set it, so the refresh worker and API handlers keep reading the
# latest data even while a backtest holds an as-of date open elsewhere.
_as_of_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "edgar_as_of", default=None
)


def as_of(date: Optional[str]):
    """
    Read the store as it stood on `date` (ISO 'YYYY-MM-DD') for this block.

    Applies to every read below, including the ones reached indirectly through
    `buffett_metrics`, so a whole ranking run can be moved back in time without
    each call site having to thread a date through.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.AsOf(date)


def _effective_as_of(explicit: Optional[str]) -> Optional[str]:
    # Ported to Rust: rust/investa_core/src/ef_store.rs
    return investa_core.ef_effective_as_of(explicit)


# --- per-company read session -------------------------------------------------
#
# Scoring one company reads its facts about ten times — the whole concept set,
# the statement subset, then one concept at a time for every split-sensitive
# series — and each read opened its own connection. Inside `company_session`
# the reads share one connection, and the two readers that repeat (the
# assembled series and the by-filing view) query each tag once: a tag's rows
# do not depend on which other tags share the query, so a later subset is
# answered from what is already loaded. The cache dies with the block, so a
# session sees the store as it stood when the company was first read, never
# across companies.


class _CompanySession:
    __slots__ = ("cik", "thread", "connections", "loaded")

    def __init__(self, cik: str):
        self.cik = cik
        # A connection belongs to the thread that opened it; a context copied
        # into another thread (asyncio.to_thread) reads without the session.
        self.thread = threading.get_ident()
        self.connections: Dict[str, sqlite3.Connection] = {}
        # (reader, db path, as_of) -> (tags loaded, {tag: rows})
        self.loaded: Dict[Tuple[str, str, Optional[str]], Tuple[set, Dict]] = {}


_session_var: contextvars.ContextVar[Optional[_CompanySession]] = (
    contextvars.ContextVar("edgar_company_session", default=None)
)


def _session_for(cik: str) -> Optional[_CompanySession]:
    # Ported to Rust: rust/investa_core/src/ef_store.rs
    return investa_core.ef_session_for(cik)


def company_session(cik: str):
    """Share one connection and each tag's rows across this block's reads of `cik`.

    A context manager (rust/investa_core/src/ef_store.rs): inside an open
    session for the same company it does nothing; on exit it resets the
    session variable and closes the session's connections.
    """
    return investa_core.CompanySession(cik)


def _db_path() -> str:
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_db_path()


class EdgarFactStore:
    """
    SQLite store of annual XBRL facts.

    Deliberately separate from `market_data.db`: this is slow-moving,
    append-heavy reference data with a different refresh cadence, and mixing it
    with the price cache would make both harder to reason about.
    """

    _write_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _db_path()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = get_db_connection(self.db_path, use_cache=False, timeout=60.0)
        if conn is None:
            raise sqlite3.OperationalError(
                f"Could not open database connection to {self.db_path}"
            )
        return conn

    def _init_db(self) -> None:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        investa_core.ef_init_db(self)

    # --- writing ----------------------------------------------------------

    def upsert_facts(self, rows: List[Tuple]) -> int:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_upsert_facts(self, rows)

    def mark_ingested(self, cik: str, fact_count: int) -> None:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        investa_core.ef_mark_ingested(self, cik, fact_count)

    def ingested_ciks(self) -> set:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_ingested_ciks(self)

    # --- quarterly writing/reading ----------------------------------------

    def upsert_quarterly_facts(self, rows: List[Tuple]) -> int:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_upsert_quarterly_facts(self, rows)

    def mark_quarterly_ingested(self, cik: str, fact_count: int) -> None:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        investa_core.ef_mark_quarterly_ingested(self, cik, fact_count)

    def quarterly_ingest_state(self, cik: str) -> Optional[Tuple[datetime, int]]:
        """
        (when this filer's quarterly facts were last loaded, how many landed).

        The count matters to the caller: an ingest that stored nothing is either
        a filer with no quarterly XBRL or a request that failed, and neither
        should be treated like a good load a week old.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_quarterly_ingest_state(self, cik)

    def get_many_tag_spans(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[str, Dict[Tuple[str, str], Tuple[float, str]]]:
        """
        {tag: {(period_start, period_end): (value, unit)}} from the quarterly
        table, taking the most recently filed value for a restated span.

        The span, not the end alone, is the key: the three-month and the
        year-to-date figure a 10-Q reports share an end date, and telling them
        apart is what makes a real quarterly series derivable. Instants (balance
        sheet) carry an empty start.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_many_tag_spans(self, cik, tags, as_of)

    def get_many_tag_spans_by_filing(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[str, List[Filing]]:
        """
        {tag: [one filing's spans, newest filed first]} — the quarterly twin of
        `get_tag_series_by_accession`.

        Spans stay with the filing that reported them instead of collapsing to
        the newest. `get_many_tag_spans` takes the newest filing *per span
        independently*, and for a split-sensitive tag that mixes two bases inside
        one fiscal year: a later 10-K restates the annual span it carries as a
        comparative, while the three quarterly spans of that same year are never
        re-filed. NVIDIA's FY2023 came out as one quarter of 25.1bn shares beside
        three of 2.5bn. Within one filing there is no such step, which is what
        makes a same-filing comparison the way to put them back on one basis.

        Ordered by the filing date, and returned already ordered, because the
        accession number cannot supply it: the prefix is the *filing agent's*
        CIK, not the filer's, and it changes when a company moves agent or starts
        filing for itself. Apple's runs 0001193125 (2009-2016), then
        0001628280, then its own 0000320193 — so sorting the accessions
        descending puts 2017 first and 2026 last.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_many_tag_spans_by_filing(self, cik, tags, as_of)

    # --- reading ----------------------------------------------------------

    def get_tag_series(
        self, cik: str, tag: str, as_of: Optional[str] = None
    ) -> Dict[str, Tuple[float, str]]:
        """
        {period_end: (value, unit)} for one tag, taking the most recently filed
        value when a period has been restated.

        With `as_of`, only facts filed on or before that date are visible, so
        the answer is the originally-reported one rather than a later restatement.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_tag_series(self, cik, tag, as_of)

    @staticmethod
    def _many_tag_series_query(cik, tag_list, as_of):
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_many_tag_series_query(cik, tag_list, as_of)

    def get_many_tag_series(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[str, Dict[str, Tuple[float, str]]]:
        """Batched `get_tag_series` — one query instead of one per tag."""
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_many_tag_series(self, cik, tags, as_of)

    def get_tag_revisions(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[Tuple[str, str], List[Tuple[str, float, str]]]:
        """
        {(tag, period_end): [(filed, value, form), ...]} filed-ascending.

        The whole filing history of each number rather than the winner. Keeping
        every revision is the reason this store never overwrites a fact, and it
        is what makes "they said 1.2bn in 2020 and 1.1bn in 2022" answerable at
        all — no vendor feed carries it, because a feed serves the current view.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_tag_revisions(self, cik, tags, as_of)

    def get_tag_series_by_filing(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        {tag: [one filing's {period_end: value}, newest filed first]} — values
        kept with the filing that reported them instead of collapsing to the
        latest.

        The default reader takes the most recently filed value per period, which
        is right for levels and wrong for anything compared *across* years: a
        10-K restates the two prior years for a stock split but nothing restates
        the years before that, so the assembled series steps by the split ratio
        at whatever year the restatements stop. Within one filing there is no
        such step, which is what makes a same-filing comparison the way to tell
        a split apart from real issuance.

        Ordered here, by the filing date, because the accession number cannot
        supply it: the prefix is the *filing agent's* CIK rather than the
        filer's, and it changes when a company switches agent or starts filing
        for itself. Apple's runs 0001193125 (2009-2016), then 0001628280, then
        its own 0000320193 — so sorting the accessions descending called a 2016
        filing the newest when the newest was from 2025, and the caller's "most
        authoritative filing first" was neither.
        """
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_get_tag_series_by_filing(self, cik, tags, as_of)

    def has_data(self, cik: str) -> bool:
        # Ported to Rust: rust/investa_core/src/ef_store.rs
        return investa_core.ef_has_data(self, cik)


_shared_store: Optional[EdgarFactStore] = None


def get_store() -> EdgarFactStore:
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_store()


# --- parsing ----------------------------------------------------------------
def _is_annual_duration(start: Optional[str], end: str) -> bool:
    """True for facts covering roughly one year; instants have no start."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_is_annual_duration(start, end)


def parse_company_facts(
    payload: Dict[str, Any], wanted_tags: Optional[set] = None
) -> List[Tuple]:
    """
    Flatten one companyfacts document into rows for the `facts` table.

    Keeps only annual facts from annual reports, and only the tags any concept
    chain actually references — that filter is what turns a 1.4 GB archive into
    a database of workable size.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_parse_company_facts(payload, wanted_tags)


def _span_days(start: Optional[str], end: str) -> Optional[int]:
    """Length of a duration fact in days, or None if it is an instant."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_span_days(start, end)


def _canonical_period_ends(
    spans_by_tag: Dict[str, Dict[Tuple[str, str], Tuple[float, str]]],
) -> Dict[str, str]:
    """
    {end as filed: the end it should be read as} across one filer's whole store.

    A period end mistyped in a single filing would otherwise become an extra
    quarter of its own. Ends within `_PERIOD_END_TOLERANCE_DAYS` of each other
    are the same period, and the one the most facts agree on is the real one: a
    mistake stays inside the filing that made it, while the true date is
    corroborated by every other filing that reported the quarter and by the
    balance-sheet instants that close on it.

    Counted across all tags rather than per tag on purpose — the instants are
    what make the corroboration lopsided, and they belong to other tags than the
    income-statement rows the typo landed on.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_canonical_period_ends(spans_by_tag)


def _apply_canonical_ends(
    spans: Dict[Tuple[str, str], Tuple[float, str]], canonical: Dict[str, str]
) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """
    Rewrite one tag's spans onto the canonical period ends.

    Where a rewrite lands on a span already filed under the canonical end, the
    one filed there wins: it is the corroborated reading, and letting the typo
    overwrite it would trade a wrong date for a wrong value.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_apply_canonical_ends(spans, canonical)


def _filing_basis_factors(filings: List[Filing]) -> Dict[Tuple[str, str], float]:
    """
    {filing: the factor putting its figures on the newest filing's split basis}.

    Taking the newest filing per span independently is what mixes bases: a 10-K
    restates the annual span it carries as a comparative, and the three quarterly
    spans inside that year are never re-filed, so NVIDIA's FY2023 arrived as one
    quarter of 25.1bn shares beside three of 2.5bn. A single per-year factor
    cannot fix that — it scales both bases by the same ratio.

    Measured per *filing*, newest first. A filing is internally consistent by
    construction, so where an older filing and an already-placed one report the
    same span, their ratio is exactly what puts the rest of that older filing on
    today's basis. The median of those ratios is taken so a genuine restatement
    of one line does not move the whole filing, and the chain composes: a filing
    overlapping no recent one but overlapping a middle one is bridged through it.

    A factor per filing rather than per span, because that is what the basis
    belongs to. Keyed by span it could not answer for a figure the share count
    never covered — Apple files a fourth-quarter EPS with no matching
    three-month share count — and that figure would keep the old basis while its
    siblings moved, which is the very mixing this is here to end.

    Filings nothing corroborates get 1.0 and keep their filed values. A filer
    whose filings simply do not overlap is left as it was rather than quietly
    reshaped, the same concession `split_consistent_series` makes annually.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_filing_basis_factors(filings)


def _median(values: List[float]) -> float:
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_median(values)


def parse_company_quarterly_facts(
    payload: Dict[str, Any], wanted_tags: Optional[set] = None
) -> List[Tuple]:
    """
    Flatten one companyfacts document into rows for the `quarterly_facts` table.

    Keeps every duration up to a full year from the periodic reports, not just
    the three-month ones, because most filers tag their cash-flow statement
    year-to-date only: Q2 exists solely as (six months − three months), and Q4
    only ever as (full year − nine months). `_derive_quarterly_series` does that
    subtraction; this just has to preserve the ingredients.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_parse_company_quarterly_facts(payload, wanted_tags)


def _derive_quarterly_series(
    spans: Dict[Tuple[str, str], Tuple[float, str]],
    additive: bool = True,
) -> Dict[str, Tuple[float, str]]:
    """
    One tag's three-month series, keyed by period end.

    Two kinds of fact go in. A duration already three months long is a quarter
    as filed and is taken at face value. Everything else is a year-to-date
    figure, and the quarters hide inside the ladder that shares its start date:
    for a December filer, `Jan–Mar`, `Jan–Jun`, `Jan–Sep`, `Jan–Dec` differenced
    step by step gives Q1..Q4. Q4 is only ever recoverable this way — no 10-Q
    covers it.

    Differencing is refused across a gap: if the nine-month figure is missing,
    (full year − six months) is two quarters, not one, and emitting it as Q4
    would be a fabricated number rather than a missing one.

    `additive=False` marks a tag that must never be differenced. A weighted
    average share count is the case: subtracting the nine-month average from
    the full-year average is not the fourth quarter's average, it is noise
    around zero — Meta's Q4 2025 came out at *minus* four million shares. For
    those, the shortest duration ending on the date wins, so Q1–Q3 are the
    filed three-month averages and Q4 falls back to the annual one, which for a
    slowly-moving level is a close stand-in and is at least a real filed figure.

    As-filed quarters win over derived ones wherever both exist.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_derive_quarterly_series(spans, additive)


# --- ingest -----------------------------------------------------------------
def download_bulk_archive(
    dest_path: Optional[str] = None, force: bool = False
) -> Optional[str]:
    """
    Download `companyfacts.zip` (~1.4 GB). Returns the local path, or None.

    Skips the download if a plausible archive already exists, since this is by
    far the most expensive step of a cold build.
    """
    if dest_path is None:
        cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "edgar")
        os.makedirs(cache_dir, exist_ok=True)
        dest_path = os.path.join(cache_dir, "companyfacts.zip")

    if (
        not force
        and os.path.exists(dest_path)
        and os.path.getsize(dest_path) > 500_000_000
    ):
        logging.info(f"EDGAR: reusing existing archive at {dest_path}")
        return dest_path

    logging.info("EDGAR: downloading companyfacts.zip (~1.4 GB, this takes a while)")

    # Streamed to a temporary file in chunks: buffering the whole archive in
    # memory would cost 1.4 GB of RSS, and a partial write must never be left
    # at the real path where the size check above would accept it as complete.
    temp_path = dest_path + ".part"
    request = urllib.request.Request(
        _BULK_URL, headers={"User-Agent": get_user_agent()}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            downloaded = 0
            with open(temp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * 1024 * 1024) < len(chunk):
                        logging.info(f"EDGAR: {downloaded / 1e6:.0f} MB downloaded")
        os.replace(temp_path, dest_path)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logging.error(f"EDGAR: bulk download failed: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

    logging.info(f"EDGAR: archive saved to {dest_path}")
    return dest_path


def iter_bulk_documents(
    zip_path: str, ciks: Optional[set] = None
) -> Iterator[Dict[str, Any]]:
    """
    Yield companyfacts documents from the archive, one at a time.

    Streaming member-by-member keeps peak memory at one JSON document rather
    than the whole 1.4 GB archive.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.BulkDocuments(zip_path, ciks)


def ingest_bulk(
    zip_path: str,
    ciks: Optional[set] = None,
    limit: Optional[int] = None,
    progress_every: int = 250,
) -> Dict[str, int]:
    """
    Load the bulk archive into the fact store.

    Restricting to `ciks` (the rankable universe) avoids storing facts for the
    ~5,000 filers that are not listed common stock.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_ingest_bulk(zip_path, ciks, limit, progress_every)


def ingest_company(cik: str) -> int:
    """Refresh a single CIK from the REST API. Returns the number of facts stored."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_ingest_company(cik)


def ingest_company_quarterly(
    cik: str,
    max_age_days: float = 7,
    empty_retry_hours: float = 1,
    timeout: int = 60,
    retries: int = 3,
) -> int:
    """
    Load one filer's quarterly facts, on demand. Returns the rows stored.

    Per company rather than in bulk: quarterly facts are read only when someone
    opens a stock's statements, and keeping every duration for 5,600 filers
    would multiply the store for data almost none of it would ever be looked at.
    One companyfacts document is a few MB and lands in about a second.

    A filer loaded within `max_age_days` is left alone — nothing new is filed
    between two visits on the same afternoon. Only a *failed request* comes back
    sooner, after `empty_retry_hours`: a week of showing five quarters is the
    wrong answer to a timeout. A filer that answered with no quarterly XBRL has
    answered, and waits out the full interval like any other — the two used to be
    indistinguishable in a row count, which had the 548 foreign issuers whose
    facts are empty re-downloading a multi-megabyte document every hour.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_ingest_company_quarterly(
        cik, max_age_days, empty_retry_hours, timeout, retries
    )


# --- concept resolution -----------------------------------------------------
def resolve_concept(
    tag_series: Dict[str, Dict[str, Tuple[float, str]]], chain: List[str]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Walk a fallback chain, period by period.

    Returns ({period_end: value}, {period_end: tag_that_answered}). Resolving per
    *period* rather than per company is what recovers history across an
    accounting-standard change: Apple's revenue comes from `Revenues` for the
    older years and `RevenueFromContractWithCustomer...` for the newer ones, in
    one continuous series.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_resolve_concept(tag_series, chain)


def get_concept_values(
    cik: str, concepts: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    {concept: {period_end: value}} for the requested concepts (default: all).
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_concept_values(cik, concepts)


def _values_by_filing(cik: str, concept: str) -> List[Dict[str, float]]:
    """
    One entry per (tag, filing) for `concept`, in the order a ratio should be
    trusted: preferred tag first, and within a tag the newest filing first.

    Grouped by tag *and* accession because mixing two tags inside one filing
    would reintroduce exactly the inconsistency the callers are ruling out. The
    ordering matters in the 0.4% of adjacent pairs where two filings disagree by
    more than a percent — a genuine restatement rather than a split basis — and
    there the newest filing is the answer, which is the same rule the rest of
    this module follows.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_values_by_filing(cik, concept)


def _filed_ratio(
    by_filing: List[Dict[str, float]], earlier: str, later: str
) -> Optional[float]:
    """The later-over-earlier ratio as the most authoritative filing reported it."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_filed_ratio(by_filing, earlier, later)


def split_consistent_series(cik: str, concept: str) -> Dict[str, float]:
    """
    A series that can be compared across years, rebuilt from same-filing ratios.

    The assembled series takes the most recently filed value for each period,
    which is right for levels and wrong for rates: a 10-K restates the two prior
    years for a stock split and nothing restates the years before it, so the
    series steps by the split ratio at whatever year the restatements stop.
    Apple's diluted share count steps 5.25bn -> 20.0bn between FY2017 and FY2018
    and reads as +11.8%/yr of issuance across a decade in which it retired a
    quarter of its shares.

    Anchored on the newest value — that one is on today's split basis — and
    chained backwards: each earlier period is set from the year-over-year ratio
    a *single filing* reported for that pair, which no split can distort because
    a filing never contradicts itself. Pairs no filing covers keep the assembled
    relationship, so a company whose filings simply do not overlap is left
    exactly as it was rather than being quietly reshaped.

    Respects the point-in-time window: under `as_of` only filings visible then
    take part, so a backtest reconstructs the series an investor could have
    built at the time.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_split_consistent_series(cik, concept)


# Concepts whose value a stock split rescales. Excluded from revision detection:
# a 4:1 split changes every prior-year share count and EPS by exactly 4x, which
# would swamp the list with events that are not revisions of anything.
_SPLIT_RESCALED_CONCEPTS = frozenset(
    {"shares_diluted", "shares_basic", "shares_outstanding", "eps_diluted", "eps_basic"}
)


def revisions(
    cik: str,
    concepts: Optional[List[str]] = None,
    min_change: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Where a later filing changed a number this company had already reported.

    Compares the first filed value for each (tag, period) against the newest one.
    Restatements are ordinary — a discontinued operation reclassifies years of
    revenue, and an error correction looks identical in the data — so this
    reports magnitude and dates and calls nothing fraud.

    Two things it deliberately does not count:

      * **Tag switches.** A company moving from `Revenues` to
        `RevenueFromContractWithCustomerExcludingAssessedTax` reports the same
        year under two names, and comparing across them would invent a revision
        out of an accounting-standard change. Comparison is always within a tag.
      * **Splits.** Share counts and per-share figures are rescaled by every
        later split, so they are excluded outright rather than reported as
        thousands of revisions of nothing.

    Only the tag that actually answers for a period is reported, so a line item
    appears once and the revision shown is a revision of the number the rest of
    the app uses. Without this Boeing's equity shows up twice for 2017 — once
    from the concept's preferred tag and once from a fallback holding a narrower
    figure — which reads as two different restatements of one line.

    `min_change` is relative; below a percent is rounding, not a restatement.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_revisions(cik, concepts, min_change)


def get_concept_provenance(
    cik: str, concepts: Optional[List[str]] = None
) -> Dict[str, Dict[str, str]]:
    """Which tag answered for each concept and period. Used by the coverage report."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_concept_provenance(cik, concepts)


# --- statement assembly -----------------------------------------------------

# Concept → yfinance row label. Emitting yfinance's vocabulary means the whole
# existing ratio/DCF/Graham stack consumes EDGAR data unchanged.
_INCOME_LABELS = {
    "revenue": "Total Revenue",
    "cost_of_revenue": "Cost Of Revenue",
    "gross_profit": "Gross Profit",
    "operating_income": "Operating Income",
    "pretax_income": "Pretax Income",
    "tax_provision": "Tax Provision",
    "net_income": "Net Income",
    "interest_expense": "Interest Expense",
    "eps_diluted": "Diluted EPS",
    "shares_diluted": "Diluted Average Shares",
    "shares_basic": "Basic Average Shares",
}

_BALANCE_LABELS = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "equity": "Stockholders Equity",
    "equity_including_minority": "Total Equity Gross Minority Interest",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
    "inventory": "Inventory",
    "cash": "Cash And Cash Equivalents",
    "shares_outstanding": "Ordinary Shares Number",
}

_CASHFLOW_LABELS = {
    "operating_cash_flow": "Operating Cash Flow",
    "depreciation_amortization": "Depreciation And Amortization",
}

# EDGAR records these as payments — positive numbers leaving the company —
# whereas yfinance reports them as negative cash-flow lines. `financial_ratios`
# computes free cash flow as `ocf + capex`, so the sign must be flipped or every
# FCF in the system doubles.
_NEGATE_ON_EMIT = {
    "capex": "Capital Expenditure",
    "dividends_paid": "Cash Dividends Paid",
    "share_repurchase": "Repurchase Of Capital Stock",
}


# Rows a stock split rescales. Shares move with the split, per-share figures move
# against it, and both are reported as filed — so a nineteen-year statement shows
# Apple earning $9.21 a share in FY2017 and $2.98 in FY2018, which reads as a
# collapse rather than the 4:1 split it is. Restated onto the latest basis so the
# column-to-column comparison a statement table invites is a real one.
_SHARE_SCALED = ("shares_diluted", "shares_basic", "shares_outstanding")
_PER_SHARE_SCALED = ("eps_diluted", "eps_basic")


def _all_chain_tags() -> List[str]:
    """Every tag a concept chain reads: what scoring one company asks for in all."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_all_chain_tags()


def _tags_for(concepts: Tuple[str, ...]) -> List[str]:
    """The tags behind some concepts, in the order their fallback chains rank them."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_tags_for(concepts)


def _quarterly_basis_factors(cik: str) -> Dict[Tuple[str, str], float]:
    """
    {filing: the factor putting its figures on the newest split basis}.

    Measured on the share count and on nothing else, which is the choice
    `split_adjustment_factors` makes annually and for the same reason: two
    filings reporting one period's *share count* differently have been through a
    split, while two filings disagreeing about earnings per share may simply have
    restated earnings. Reading the second as a split rescaled every quarter of
    Microsoft's FY2011–FY2015 by 1.21, against filed annual figures that were
    already right.

    Sharing one factor between the share rows and the per-share rows is also what
    keeps `EPS x shares = net income` true across the table.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_quarterly_basis_factors(cik)


def _spans_on_latest_basis(
    filings: List[Filing],
    factors: Dict[Tuple[str, str], float],
    inverse: bool = False,
) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """
    One tag's spans, each taken from the newest filing that reported it and put
    on the latest basis by that filing's own factor.

    Selecting the value and correcting it in one pass is what keeps the two in
    step: a value read from one filing and scaled by another filing's factor is
    on neither basis. Shares move with a split and per-share figures against it,
    so `inverse` divides where the share rows multiply.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_spans_on_latest_basis(filings, factors, inverse)


def split_adjustment_factors(cik: str) -> Dict[str, float]:
    """
    {period_end: latest-basis factor} derived from the share count.

    One factor for every rescaled row rather than a reconstruction per concept:
    a split moves shares and EPS by the same ratio in opposite directions, so
    sharing the factor is what keeps `EPS x shares = net income` true down the
    whole table. The newest period's factor is 1.0 by construction — that is the
    basis everything is restated onto.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_split_adjustment_factors(cik)


def _apply_split_adjustment(
    values: Dict[str, Dict[str, float]], factors: Dict[str, float]
) -> None:
    """
    Rescale the split-sensitive concepts in place.

    Periods the diluted share count does not cover keep their filed value: with
    no factor there is nothing to restate onto, and inventing one would be worse
    than a visible step.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_apply_split_adjustment(values, factors)


def concept_labels() -> Dict[str, str]:
    """
    Human label for each statement concept, in yfinance's vocabulary.

    The same names the statements tab prints, so a revision reads as "Total
    Revenue" rather than `revenue` and lines up with the row a reader can go and
    look at. Includes the sign-flipped concepts — `capex` is still "Capital
    Expenditure" whichever way the cash flows.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_concept_labels()


def _frame_from_concepts(
    values: Dict[str, Dict[str, float]],
    labels: Dict[str, str],
    negate: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Build a yfinance-shaped frame: rows are line items, columns are periods."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_frame_from_concepts(values, labels, negate)


def get_statements(cik: str) -> Dict[str, pd.DataFrame]:
    """
    Income statement, balance sheet and cash-flow frames for one company.

    Shape and labels match yfinance's `Ticker.financials` / `.balance_sheet` /
    `.cashflow`, so these can be passed straight to
    `financial_ratios.calculate_key_ratios_timeseries` and the valuation models.

    Share counts and per-share figures are restated onto the latest split basis;
    everything else is exactly as filed. Dollar totals are untouched, and the
    newest period is its own basis, so the columns the valuation models read
    (they take the latest one) are unchanged.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_statements(cik)


def get_quarterly_concept_values(
    cik: str, concepts: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    {concept: {quarter_end: value}} — the quarterly twin of `get_concept_values`.

    Each tag's three-month series is derived first, then the same fallback chain
    is walked period by period, so a concept that changed tag mid-decade still
    reads as one continuous series.

    Share counts and per-share figures come back on the newest filing's split
    basis, rebased before anything is derived — a quarter differenced out of two
    filings that disagree about the basis is not a quarter at all.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_quarterly_concept_values(cik, concepts)


def get_quarterly_statements(cik: str) -> Dict[str, pd.DataFrame]:
    """
    Income statement, balance sheet and cash-flow frames by fiscal quarter.

    Same shape and labels as `get_statements`, so every consumer of the annual
    frames reads these unchanged. Income and cash-flow rows are three-month
    figures — differenced out of the year-to-date ladder where the filer only
    tagged it that way — and balance-sheet rows are the quarter-end instants as
    filed.

    No split adjustment is applied here, unlike `get_statements`: the share and
    per-share spans were rebased onto the newest filing's basis in
    `get_quarterly_concept_values`, before the differencing that a mixed basis
    would have corrupted. Rescaling the derived series again would double it.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_quarterly_statements(cik)


def _add_derived_rows(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    values: Dict[str, Dict[str, float]],
    cashflow: Optional[pd.DataFrame] = None,
) -> None:
    """
    Fill in line items EDGAR does not tag directly but the ratio engine expects.

    Kept separate from `_frame_from_concepts` because these are *derivations*,
    not reported facts, and the distinction matters when auditing a score.

    Each line is written once (`_write_row`) rather than one `.loc` cell at a
    time, each of which could enlarge the frame. That was a fifth of the
    ranking batch.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_add_derived_rows(income, balance, values, cashflow)


_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _period_stamps(values: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """`pd.to_datetime(period_end, errors="coerce")` for every ISO-day period
    end in `values`, parsed together. A single string costs pandas ~0.1 ms of
    format guessing; for `YYYY-MM-DD` strings the vectorised parse gives the
    same Timestamp (NaT for an impossible day). Other strings are left to the
    caller's one-at-a-time parse."""
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_period_stamps(values)


def _write_row(frame: pd.DataFrame, label: str, cells: list) -> None:
    """`frame.loc[label, column] = value` for each cell in order, in one write.

    A later cell for the same column wins, as it would cell by cell. A new
    label is appended as one row, NaN where no period has a value. Frames this
    does not cover exactly (anything but unique float64 columns, or a label
    already present more than once) are written cell by cell.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_write_row(frame, label, cells)


def get_sector_concepts(cik: str, sector_model: str) -> Dict[str, Dict[str, float]]:
    """
    Sector-specific inputs for the bank and REIT models (Stage 3b).

    Returns {} for an unknown model rather than raising, so a caller iterating a
    mixed universe never needs to branch defensively.
    """
    # Ported to Rust: rust/investa_core/src/ep.rs
    return investa_core.ep_get_sector_concepts(cik, sector_model)
