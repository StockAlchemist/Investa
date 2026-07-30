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

import contextlib
import contextvars
import io
import json
import logging
import os
import sqlite3
import threading
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

import config
from edgar_concepts import (
    BALANCE_CONCEPTS,
    BANK_CONCEPTS,
    CASHFLOW_CONCEPTS,
    INCOME_CONCEPTS,
    REIT_CONCEPTS,
    all_concepts,
    all_tags,
)
from edgar_http import get_user_agent, sec_get_json

_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Only annual reports. Quarterlies would let a company's fiscal calendar leak
# into cross-sectional comparisons, and every metric here is annual by design.
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})

# A "duration" fact is annual if it spans roughly a year. Filers are not exact:
# 52/53-week retail calendars produce 364- and 371-day years.
_MIN_ANNUAL_DAYS = 300
_MAX_ANNUAL_DAYS = 400

_DB_FILENAME = "edgar_facts.db"

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


@contextlib.contextmanager
def as_of(date: Optional[str]):
    """
    Read the store as it stood on `date` (ISO 'YYYY-MM-DD') for this block.

    Applies to every read below, including the ones reached indirectly through
    `buffett_metrics`, so a whole ranking run can be moved back in time without
    each call site having to thread a date through.
    """
    token = _as_of_var.set(date)
    try:
        yield
    finally:
        _as_of_var.reset(token)


def _effective_as_of(explicit: Optional[str]) -> Optional[str]:
    return explicit if explicit is not None else _as_of_var.get()


def _db_path() -> str:
    directory = os.path.join(config.get_app_data_dir(), config.DB_DIR)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _DB_FILENAME)


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
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    cik TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    period_start TEXT,
                    val REAL,
                    unit TEXT,
                    form TEXT,
                    accn TEXT NOT NULL,
                    filed TEXT,
                    PRIMARY KEY (cik, tag, period_end, accn)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_cik_tag ON facts (cik, tag)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_log (
                    cik TEXT PRIMARY KEY,
                    ingested_at TEXT,
                    fact_count INTEGER
                )
            """)
            conn.commit()

    # --- writing ----------------------------------------------------------

    def upsert_facts(self, rows: List[Tuple]) -> int:
        if not rows:
            return 0
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO facts
                   (cik, tag, period_end, period_start, val, unit, form, accn, filed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        return len(rows)

    def mark_ingested(self, cik: str, fact_count: int) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ingest_log (cik, ingested_at, fact_count) VALUES (?, ?, ?)",
                (cik, datetime.now().isoformat(), fact_count),
            )
            conn.commit()

    def ingested_ciks(self) -> set:
        with self._connect() as conn:
            return {row[0] for row in conn.execute("SELECT cik FROM ingest_log")}

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
        as_of = _effective_as_of(as_of)
        query = """
            SELECT period_end, val, unit, filed FROM facts
            WHERE cik = ? AND tag = ?
        """
        params: List[Any] = [cik, tag]
        if as_of:
            query += " AND filed <= ?"
            params.append(as_of)
        query += " ORDER BY period_end, filed"

        result: Dict[str, Tuple[float, str]] = {}
        with self._connect() as conn:
            for period_end, val, unit, _filed in conn.execute(query, params):
                if val is not None:
                    # Rows arrive filed-ascending, so the last write wins.
                    result[period_end] = (val, unit)
        return result

    def get_many_tag_series(
        self, cik: str, tags: Iterable[str], as_of: Optional[str] = None
    ) -> Dict[str, Dict[str, Tuple[float, str]]]:
        """Batched `get_tag_series` — one query instead of one per tag."""
        tag_list = list(tags)
        if not tag_list:
            return {}
        as_of = _effective_as_of(as_of)
        placeholders = ",".join("?" * len(tag_list))
        query = f"""
            SELECT tag, period_end, val, unit FROM facts
            WHERE cik = ? AND tag IN ({placeholders})
        """
        params: List[Any] = [cik, *tag_list]
        if as_of:
            query += " AND filed <= ?"
            params.append(as_of)
        query += " ORDER BY period_end, filed"

        result: Dict[str, Dict[str, Tuple[float, str]]] = {}
        with self._connect() as conn:
            for tag, period_end, val, unit in conn.execute(query, params):
                if val is not None:
                    result.setdefault(tag, {})[period_end] = (val, unit)
        return result

    def has_data(self, cik: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM facts WHERE cik = ? LIMIT 1", (cik,)).fetchone()
        return row is not None


_shared_store: Optional[EdgarFactStore] = None


def get_store() -> EdgarFactStore:
    global _shared_store
    if _shared_store is None:
        _shared_store = EdgarFactStore()
    return _shared_store


# --- parsing ----------------------------------------------------------------


def _is_annual_duration(start: Optional[str], end: str) -> bool:
    """True for facts covering roughly one year; instants have no start."""
    if not start:
        return True  # instant (balance-sheet) fact
    try:
        span = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    except (ValueError, TypeError):
        return False
    return _MIN_ANNUAL_DAYS <= span <= _MAX_ANNUAL_DAYS


def parse_company_facts(payload: Dict[str, Any], wanted_tags: Optional[set] = None) -> List[Tuple]:
    """
    Flatten one companyfacts document into rows for the `facts` table.

    Keeps only annual facts from annual reports, and only the tags any concept
    chain actually references — that filter is what turns a 1.4 GB archive into
    a database of workable size.
    """
    if wanted_tags is None:
        wanted_tags = set(all_tags())

    try:
        cik = str(payload["cik"]).zfill(10)
    except (KeyError, TypeError, ValueError):
        return []

    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    rows: List[Tuple] = []
    for tag, tag_body in us_gaap.items():
        if tag not in wanted_tags:
            continue
        for unit, entries in tag_body.get("units", {}).items():
            for entry in entries:
                if entry.get("form") not in _ANNUAL_FORMS:
                    continue
                end = entry.get("end")
                if not end:
                    continue
                start = entry.get("start")
                if not _is_annual_duration(start, end):
                    continue
                rows.append(
                    (
                        cik,
                        tag,
                        end,
                        start,
                        entry.get("val"),
                        unit,
                        entry.get("form"),
                        entry.get("accn", ""),
                        entry.get("filed"),
                    )
                )
    return rows


# --- ingest -----------------------------------------------------------------


def download_bulk_archive(dest_path: Optional[str] = None, force: bool = False) -> Optional[str]:
    """
    Download `companyfacts.zip` (~1.4 GB). Returns the local path, or None.

    Skips the download if a plausible archive already exists, since this is by
    far the most expensive step of a cold build.
    """
    if dest_path is None:
        cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "edgar")
        os.makedirs(cache_dir, exist_ok=True)
        dest_path = os.path.join(cache_dir, "companyfacts.zip")

    if not force and os.path.exists(dest_path) and os.path.getsize(dest_path) > 500_000_000:
        logging.info(f"EDGAR: reusing existing archive at {dest_path}")
        return dest_path

    logging.info("EDGAR: downloading companyfacts.zip (~1.4 GB, this takes a while)")

    # Streamed to a temporary file in chunks: buffering the whole archive in
    # memory would cost 1.4 GB of RSS, and a partial write must never be left
    # at the real path where the size check above would accept it as complete.
    temp_path = dest_path + ".part"
    request = urllib.request.Request(_BULK_URL, headers={"User-Agent": get_user_agent()})
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


def iter_bulk_documents(zip_path: str, ciks: Optional[set] = None) -> Iterator[Dict[str, Any]]:
    """
    Yield companyfacts documents from the archive, one at a time.

    Streaming member-by-member keeps peak memory at one JSON document rather
    than the whole 1.4 GB archive.
    """
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not info.filename.startswith("CIK") or not info.filename.endswith(".json"):
                continue
            if ciks is not None:
                # Filenames are CIK##########.json
                cik = info.filename[3:-5].zfill(10)
                if cik not in ciks:
                    continue
            try:
                with archive.open(info) as handle:
                    yield json.load(io.TextIOWrapper(handle, encoding="utf-8"))
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logging.warning(f"EDGAR: skipping {info.filename}: {exc}")


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
    store = get_store()
    stats = {"companies": 0, "facts": 0, "empty": 0}
    batch: List[Tuple] = []
    wanted = set(all_tags())

    for document in iter_bulk_documents(zip_path, ciks=ciks):
        rows = parse_company_facts(document, wanted_tags=wanted)
        cik = str(document.get("cik", "")).zfill(10)

        if rows:
            batch.extend(rows)
            stats["facts"] += len(rows)
        else:
            stats["empty"] += 1

        store.mark_ingested(cik, len(rows))
        stats["companies"] += 1

        if len(batch) >= 50_000:
            store.upsert_facts(batch)
            batch = []

        if stats["companies"] % progress_every == 0:
            logging.info(
                f"EDGAR: ingested {stats['companies']} companies, {stats['facts']} facts"
            )
        if limit and stats["companies"] >= limit:
            break

    if batch:
        store.upsert_facts(batch)

    logging.info(
        f"EDGAR: bulk ingest complete — {stats['companies']} companies, "
        f"{stats['facts']} facts, {stats['empty']} with no usable data"
    )
    return stats


def ingest_company(cik: str) -> int:
    """Refresh a single CIK from the REST API. Returns the number of facts stored."""
    cik = str(cik).zfill(10)
    payload = sec_get_json(_COMPANY_FACTS_URL.format(cik=cik))
    if not payload:
        return 0
    rows = parse_company_facts(payload)
    store = get_store()
    store.upsert_facts(rows)
    store.mark_ingested(cik, len(rows))
    return len(rows)


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
    values: Dict[str, float] = {}
    provenance: Dict[str, str] = {}

    for tag in chain:
        series = tag_series.get(tag)
        if not series:
            continue
        for period_end, (value, _unit) in series.items():
            if period_end not in values:
                values[period_end] = value
                provenance[period_end] = tag
    return values, provenance


def get_concept_values(cik: str, concepts: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    {concept: {period_end: value}} for the requested concepts (default: all).
    """
    chains = all_concepts()
    wanted = concepts or list(chains)
    tags_needed = {tag for name in wanted for tag in chains.get(name, [])}

    tag_series = get_store().get_many_tag_series(cik, tags_needed)

    resolved: Dict[str, Dict[str, float]] = {}
    for name in wanted:
        values, _prov = resolve_concept(tag_series, chains.get(name, []))
        if values:
            resolved[name] = values
    return resolved


def get_concept_provenance(cik: str, concepts: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
    """Which tag answered for each concept and period. Used by the coverage report."""
    chains = all_concepts()
    wanted = concepts or list(chains)
    tags_needed = {tag for name in wanted for tag in chains.get(name, [])}
    tag_series = get_store().get_many_tag_series(cik, tags_needed)

    provenance: Dict[str, Dict[str, str]] = {}
    for name in wanted:
        _values, prov = resolve_concept(tag_series, chains.get(name, []))
        if prov:
            provenance[name] = prov
    return provenance


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


def _frame_from_concepts(
    values: Dict[str, Dict[str, float]],
    labels: Dict[str, str],
    negate: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Build a yfinance-shaped frame: rows are line items, columns are periods."""
    columns: Dict[str, Dict[str, float]] = {}

    def add(concept: str, label: str, sign: int) -> None:
        for period_end, value in values.get(concept, {}).items():
            columns.setdefault(period_end, {})[label] = value * sign

    for concept, label in labels.items():
        add(concept, label, 1)
    for concept, label in (negate or {}).items():
        add(concept, label, -1)

    if not columns:
        return pd.DataFrame()

    frame = pd.DataFrame(columns)
    frame.columns = pd.to_datetime(frame.columns, errors="coerce")
    frame = frame.loc[:, frame.columns.notna()]
    # yfinance orders newest-first, and `financial_ratios` reads columns[0] as
    # the latest period.
    return frame.sort_index(axis=1, ascending=False)


def get_statements(cik: str) -> Dict[str, pd.DataFrame]:
    """
    Income statement, balance sheet and cash-flow frames for one company.

    Shape and labels match yfinance's `Ticker.financials` / `.balance_sheet` /
    `.cashflow`, so these can be passed straight to
    `financial_ratios.calculate_key_ratios_timeseries` and the valuation models.
    """
    concept_names = (
        list(INCOME_CONCEPTS) + list(BALANCE_CONCEPTS) + list(CASHFLOW_CONCEPTS)
    )
    values = get_concept_values(cik, concept_names)

    income = _frame_from_concepts(values, _INCOME_LABELS)
    balance = _frame_from_concepts(values, _BALANCE_LABELS)
    cashflow = _frame_from_concepts(values, _CASHFLOW_LABELS, negate=_NEGATE_ON_EMIT)

    _add_derived_rows(income, balance, values)

    return {"financials": income, "balance_sheet": balance, "cashflow": cashflow}


def _add_derived_rows(
    income: pd.DataFrame, balance: pd.DataFrame, values: Dict[str, Dict[str, float]]
) -> None:
    """
    Fill in line items EDGAR does not tag directly but the ratio engine expects.

    Kept separate from `_frame_from_concepts` because these are *derivations*,
    not reported facts, and the distinction matters when auditing a score.
    """
    # Total Debt is rarely tagged as such; it is short-term plus long-term.
    short_term = values.get("short_term_debt", {})
    long_term = values.get("long_term_debt", {})
    debt_periods = set(short_term) | set(long_term)
    if debt_periods and not balance.empty:
        for period_end in debt_periods:
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in balance.columns:
                continue
            total = short_term.get(period_end, 0.0) + long_term.get(period_end, 0.0)
            balance.loc["Total Debt", stamp] = total

    # EBIT: pretax income plus interest expense is closer to the definition than
    # operating income, which excludes non-operating items inconsistently.
    pretax = values.get("pretax_income", {})
    interest = values.get("interest_expense", {})
    if pretax and not income.empty:
        for period_end, pretax_value in pretax.items():
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in income.columns:
                continue
            income.loc["Ebit", stamp] = pretax_value + interest.get(period_end, 0.0)

    # Some filers omit Liabilities entirely; assets minus equity recovers it.
    assets = values.get("total_assets", {})
    equity = values.get("equity", {})
    if not balance.empty and "Total Liabilities Net Minority Interest" not in balance.index:
        for period_end, asset_value in assets.items():
            if period_end not in equity:
                continue
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in balance.columns:
                continue
            balance.loc["Total Liabilities Net Minority Interest", stamp] = (
                asset_value - equity[period_end]
            )


def get_sector_concepts(cik: str, sector_model: str) -> Dict[str, Dict[str, float]]:
    """
    Sector-specific inputs for the bank and REIT models (Stage 3b).

    Returns {} for an unknown model rather than raising, so a caller iterating a
    mixed universe never needs to branch defensively.
    """
    if sector_model == "bank":
        return get_concept_values(cik, list(BANK_CONCEPTS))
    if sector_model == "reit":
        return get_concept_values(cik, list(REIT_CONCEPTS))
    return {}
