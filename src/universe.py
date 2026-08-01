# -*- coding: utf-8 -*-
"""
The investable US equity universe.

Builds the list of US-listed *common stock* from the authoritative NASDAQ Trader
symbol directories, then attaches each name's SEC CIK so fundamentals can be
pulled from EDGAR.

Why these sources: NASDAQ Trader publishes every symbol listed on every US
exchange (nasdaqlisted.txt for NASDAQ, otherlisted.txt for NYSE/AMEX/ARCA and
the rest), which is the only free, complete, daily-updated listing. Index
membership lists (S&P 500 etc.) cover ~95% of market cap but only ~2,900 names —
this module deliberately covers the whole market instead.

The raw files contain far more than common stock: ETFs, warrants, units, rights,
preferred shares, depositary receipts on preferreds, and exchange test issues.
`_is_common_stock` strips those, taking ~13,000 raw rows down to ~5,500 real
companies. Filtering is conservative in the sense that it prefers dropping an
ambiguous instrument over admitting a non-equity into a ranking that assumes
common-stock economics.

Symbols are normalised to the NASDAQ Symbol convention (BRK-B, not BRK.A),
which is what yfinance expects and what the rest of this codebase already uses.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import config
from edgar_http import sec_get_json

_NASDAQ_TRADER_BASE = "https://www.nasdaqtrader.com/dynamic/SymDir/"
_SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

_CACHE_SUBDIR = "universe"
_CACHE_FILENAME = "us_common_stock.json"

# Instrument types that are not common stock. Matched against the security
# name, which is the only reliable discriminator NASDAQ Trader gives us —
# the file has no instrument-type column.
_NON_COMMON_PATTERNS = re.compile(
    r"\b("
    r"warrant(s)?|"
    r"right(s)?|"
    r"unit(s)?|"
    r"preferred|"
    r"preference|"
    r"depositary|"
    r"debenture|"
    r"note(s)? due|"
    r"subordinated|"
    r"convertible bond|"
    r"trust preferred|"
    r"capital securities|"
    r"income securities|"
    r"contingent value"
    r")\b",
    re.IGNORECASE,
)

# A coupon in the security name (e.g. "6.375% Series D") always denotes a
# fixed-income-like instrument, never common stock.
_COUPON_PATTERN = re.compile(r"\d+\.?\d*\s*%")

# Instrument suffixes encoded in the symbol itself. These catch rows whose
# security name is too terse to classify — NYSE warrants arrive as "AMPX+"
# in the NASDAQ Symbol column and "AMPX.WS" in the ACT column.
_SYMBOL_SUFFIXES = (".WS", ".WD", ".U", ".R", ".RT", ".W")


def _normalise_symbol(symbol: str) -> str:
    """
    Convert an exchange symbol to the '-' class convention.

    NASDAQ Trader is inconsistent: preferred series arrive already dashed
    (ABR-D) but share classes keep a dot (BRK.B). yfinance and the SEC ticker
    map both use dashes, so we normalise here rather than at every lookup.
    """
    return symbol.strip().upper().replace(".", "-")


@dataclass
class UniverseEntry:
    """One common-stock listing."""

    symbol: str  # yfinance convention, e.g. "BRK-B"
    name: str
    exchange: str
    cik: Optional[str] = None  # zero-padded 10-digit, e.g. "0000320193"

    @property
    def has_filings(self) -> bool:
        return bool(self.cik)


def _fetch_symbol_file(filename: str, timeout: int = 60) -> List[str]:
    """Fetch one NASDAQ Trader pipe-delimited file. Returns [] on failure."""
    url = _NASDAQ_TRADER_BASE + filename
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logging.error(f"Universe: could not fetch {filename}: {exc}")
        return []

    lines = text.splitlines()
    if not lines:
        logging.error(f"Universe: {filename} was empty")
        return []

    # The last line is a "File Creation Time: ..." trailer, not a record.
    return [ln for ln in lines[1:] if ln and not ln.startswith("File Creation Time")]


def _is_common_stock(name: str, act_symbol: str, is_etf: str, is_test: str) -> bool:
    """
    Decide whether a listing row represents common stock.

    `act_symbol` carries a '$' for preferred/derivative series (ABR$D), which is
    a stronger signal than the name and catches rows whose names are truncated.
    """
    if is_etf.strip().upper() == "Y":
        return False
    if is_test.strip().upper() == "Y":
        return False
    if "$" in act_symbol or "+" in act_symbol:
        return False
    if act_symbol.upper().endswith(_SYMBOL_SUFFIXES):
        return False
    if _NON_COMMON_PATTERNS.search(name):
        return False
    if _COUPON_PATTERN.search(name):
        return False
    return True


def _parse_nasdaq_listed(lines: List[str]) -> List[UniverseEntry]:
    """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot|ETF|NextShares"""
    entries = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 8:
            continue
        symbol, name, _category, test_issue, _status, _lot, etf, _next = parts[:8]
        if not _is_common_stock(name, symbol, etf, test_issue):
            continue
        entries.append(
            UniverseEntry(
                symbol=_normalise_symbol(symbol), name=name.strip(), exchange="NASDAQ"
            )
        )
    return entries


def _parse_other_listed(lines: List[str]) -> List[UniverseEntry]:
    """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot|Test Issue|NASDAQ Symbol"""
    exchange_names = {
        "A": "NYSE MKT",
        "N": "NYSE",
        "P": "NYSE ARCA",
        "Z": "BATS",
        "V": "IEX",
    }
    entries = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 8:
            continue
        act_symbol, name, exchange, _cqs, etf, _lot, test_issue, nasdaq_symbol = parts[
            :8
        ]
        if not _is_common_stock(name, act_symbol, etf, test_issue):
            continue
        # Prefer the NASDAQ Symbol column: it already dashes preferred series,
        # whereas CQS uses a 'p' infix that resolves nowhere.
        symbol = _normalise_symbol(nasdaq_symbol or act_symbol)
        if not symbol:
            continue
        entries.append(
            UniverseEntry(
                symbol=symbol,
                name=name.strip(),
                exchange=exchange_names.get(exchange.strip(), exchange.strip()),
            )
        )
    return entries


def get_cik_map() -> Dict[str, str]:
    """
    Map ticker → zero-padded 10-digit CIK from the SEC's own ticker file.

    Returns {} on failure; callers must treat a missing CIK as "no fundamentals
    available" rather than as an error, since foreign private issuers and some
    recent listings legitimately have no mapping.
    """
    payload = sec_get_json(_SEC_TICKER_MAP_URL)
    if not payload:
        logging.error("Universe: could not fetch SEC ticker→CIK map")
        return {}

    # The file is a dict keyed by row index: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}}
    mapping: Dict[str, str] = {}
    rows = payload.values() if isinstance(payload, dict) else payload
    for row in rows:
        try:
            ticker = str(row["ticker"]).strip().upper()
            cik = str(row["cik_str"]).zfill(10)
        except (KeyError, TypeError, AttributeError):
            continue
        if ticker:
            mapping[ticker] = cik
    logging.info(f"Universe: loaded {len(mapping)} ticker→CIK mappings from SEC")
    return mapping


def _cache_path() -> str:
    directory = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, _CACHE_SUBDIR)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _CACHE_FILENAME)


def _load_cache(max_age_days: int) -> Optional[List[UniverseEntry]]:
    path = _cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as handle:
            blob = json.load(handle)
        built_at = datetime.fromisoformat(blob["built_at"])
        if (datetime.now() - built_at).days > max_age_days:
            return None
        return [UniverseEntry(**row) for row in blob["entries"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.warning(f"Universe: ignoring unreadable cache: {exc}")
        return None


def _save_cache(entries: List[UniverseEntry]) -> None:
    try:
        with open(_cache_path(), "w") as handle:
            json.dump(
                {
                    "built_at": datetime.now().isoformat(),
                    "entries": [asdict(e) for e in entries],
                },
                handle,
            )
    except OSError as exc:
        logging.warning(f"Universe: could not write cache: {exc}")


def build_universe(attach_cik: bool = True) -> List[UniverseEntry]:
    """Fetch and filter the listing files. Always hits the network."""
    nasdaq = _parse_nasdaq_listed(_fetch_symbol_file("nasdaqlisted.txt"))
    other = _parse_other_listed(_fetch_symbol_file("otherlisted.txt"))

    # A handful of symbols appear in both files; first occurrence wins.
    by_symbol: Dict[str, UniverseEntry] = {}
    for entry in nasdaq + other:
        by_symbol.setdefault(entry.symbol, entry)

    entries = sorted(by_symbol.values(), key=lambda e: e.symbol)
    logging.info(f"Universe: {len(entries)} common-stock listings after filtering")

    if attach_cik:
        cik_map = get_cik_map()
        if cik_map:
            matched = 0
            for entry in entries:
                # SEC keys on the plain ticker; its file uses '-' for classes too.
                cik = cik_map.get(entry.symbol.upper())
                if cik:
                    entry.cik = cik
                    matched += 1
            logging.info(
                f"Universe: matched {matched}/{len(entries)} symbols to a CIK "
                f"({len(entries) - matched} have no SEC filings and cannot be ranked)"
            )

    return entries


def get_universe(
    force_refresh: bool = False, max_age_days: int = 7
) -> List[UniverseEntry]:
    """
    The cached universe, rebuilt if older than `max_age_days`.

    Listings change slowly, so a weekly rebuild is ample; the cost of a stale
    entry is one unrankable symbol, not a wrong rank.
    """
    if not force_refresh:
        cached = _load_cache(max_age_days)
        if cached:
            logging.info(f"Universe: {len(cached)} listings from cache")
            return cached

    entries = build_universe()
    if entries:
        _save_cache(entries)
    return entries


_CACHED_CIK_MAP: Optional[Dict[str, str]] = None


def get_cached_cik_map() -> Dict[str, str]:
    """
    Ticker → CIK read from the on-disk universe cache, never from the network.

    For callers on a request path, where a cache miss must cost nothing: a
    symbol the last universe build did not cover simply has no filed history to
    show, which is the same answer a foreign private issuer gets. Use
    `get_cik_map()` when a live fetch is acceptable.

    The cache is read once per process; listings change on the order of days,
    and the rebuild that refreshes them writes the file this reads.
    """
    global _CACHED_CIK_MAP
    if _CACHED_CIK_MAP is None:
        # Age is not a reason to reject here: a stale CIK is still the right CIK
        # (they are permanent), and there is no fallback to fall back to.
        entries = _load_cache(max_age_days=36500) or []
        _CACHED_CIK_MAP = {e.symbol.upper(): e.cik for e in entries if e.cik}
    return _CACHED_CIK_MAP


def get_rankable_universe(force_refresh: bool = False) -> List[UniverseEntry]:
    """Universe members that have an SEC CIK, i.e. those we can pull fundamentals for."""
    return [e for e in get_universe(force_refresh=force_refresh) if e.has_filings]


def group_by_cik(entries: List[UniverseEntry]) -> Dict[str, List[UniverseEntry]]:
    """
    Group listings by CIK so multi-class companies can be collapsed.

    ~47 companies list more than one class (BRK-A/BRK-B, GOOG/GOOGL, AGM/AGM-A).
    They share one set of financial statements, so ranking each class separately
    would double-count the same business. Callers should keep a single class —
    normally the most liquid, which needs market data this module does not have.
    """
    grouped: Dict[str, List[UniverseEntry]] = {}
    for entry in entries:
        if entry.cik:
            grouped.setdefault(entry.cik, []).append(entry)
    return grouped
