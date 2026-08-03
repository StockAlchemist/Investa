# -*- coding: utf-8 -*-
"""
SIC industry codes, and the valuation model each company is routed to.

Sector routing is not cosmetic here — it decides *which valuation model a
company is scored by*, and scoring a bank with a free-cash-flow DCF produces a
confident, meaningless number. Three models exist:

  * `generic` — FCF/DCF plus the standard quality pillars.
  * `bank`    — residual income on book value; FCF is not a coherent quantity
                for a deposit-taking institution, and leverage ratios that
                condemn an industrial are normal for a bank.
  * `insurer` — also residual income, but scored without the lending metrics
                (net interest margin, loan-loss provisioning) an insurer has no
                reason to report.
  * `reit`    — FFO/AFFO and NAV; GAAP earnings understate REIT economics
                because depreciation is charged on assets that appreciate.

SIC is used rather than yfinance's `sector` string because the distinctions that
matter here are exactly the ones yfinance blurs: its "Financial Services" bucket
mixes deposit-taking banks (residual income) with asset managers and exchanges
(ordinary FCF businesses), and its "Real Estate" bucket mixes REITs with
brokerages. SIC 6798 means REIT; SIC 6021 means national commercial bank.

Source is the SEC's own DERA "Financial Statement Data Sets": one ~130 MB zip
per quarter whose `sub.txt` carries cik→sic for every filer that quarter. A few
quarters cover effectively the whole active universe, which is far cheaper than
the 5 MB-per-company submissions API (~28 GB for this universe).
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from datetime import date
from typing import Dict, List, Optional

import config
from edgar_http import sec_get

_DERA_URL = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
)
_CACHE_FILENAME = "cik_sic.json"

# --- SIC ranges -------------------------------------------------------------
# Depository institutions and insurers. Both are scored on residual income:
# their liabilities are operating inputs, not financing, so debt/equity and
# interest coverage are meaningless as quality gates.
_BANK_RANGES = (
    (6020, 6036),  # national, state, savings banks
    (6060, 6062),  # credit unions
    (6080, 6082),  # foreign bank branches
    (6110, 6111),  # federal credit agencies
    (6120, 6120),  # savings institutions
)

# Insurers are financials but not lenders. Measured coverage showed that ~29% of
# everything routed to the bank model had no deposits, loan book or net interest
# income — those were the insurers, and scoring them on lending metrics would
# have demoted every one of them for lacking data they will never report.
# They share the residual-income valuation but carry their own metric set.
_INSURER_RANGES = ((6311, 6411),)

# Deliberately NOT financials for our purposes: asset managers, brokers and
# finance-services companies (6199, 6211) earn fees and have ordinary
# free-cash-flow economics, so the generic model fits them better than a
# book-value model does.

# 6798 is the dedicated REIT code. 6500–6552 are real-estate operators and
# developers, which are *not* REITs — they keep the generic model.
_REIT_RANGES = ((6798, 6798),)


def _in_ranges(sic: int, ranges) -> bool:
    return any(low <= sic <= high for low, high in ranges)


def model_for_sic(sic: Optional[int]) -> str:
    """Map a SIC code to a valuation model. Unknown codes fall back to generic."""
    if sic is None:
        return "generic"
    if _in_ranges(sic, _REIT_RANGES):
        return "reit"
    if _in_ranges(sic, _INSURER_RANGES):
        return "insurer"
    if _in_ranges(sic, _BANK_RANGES):
        return "bank"
    return "generic"


def recent_quarters(count: int = 8, today: Optional[date] = None) -> List[str]:
    """
    The last `count` completed SEC filing quarters, newest first, as '2025q3'.

    Several quarters are needed because a filer only appears in the quarters it
    actually filed in; annual-only filers would be missed by a single quarter.
    Four quarters left ~13% of the ranking universe unmapped — and an unmapped
    bank is silently scored as an industrial, which is a real mis-ranking rather
    than a missing one. Eight quarters is a one-off cost paid to avoid that.
    """
    today = today or date.today()
    quarter = (today.month - 1) // 3 + 1
    year = today.year

    quarters = []
    for _ in range(count):
        # Step back one quarter first: the current one is usually incomplete.
        quarter -= 1
        if quarter < 1:
            quarter = 4
            year -= 1
        quarters.append(f"{year}q{quarter}")
    return quarters


def _parse_sub_file(payload: bytes) -> Dict[str, int]:
    """Extract cik→sic from one quarterly archive's sub.txt."""
    mapping: Dict[str, int] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            raw = archive.read("sub.txt").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        logging.error(f"SIC: could not read sub.txt: {exc}")
        return mapping

    lines = raw.splitlines()
    if not lines:
        return mapping

    header = lines[0].split("\t")
    try:
        cik_index = header.index("cik")
        sic_index = header.index("sic")
    except ValueError:
        logging.error("SIC: sub.txt is missing cik or sic column")
        return mapping

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(cik_index, sic_index):
            continue
        try:
            cik = str(int(parts[cik_index])).zfill(10)
            sic = int(parts[sic_index])
        except (ValueError, TypeError):
            continue
        mapping[cik] = sic
    return mapping


def _cache_path() -> str:
    directory = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "edgar")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _CACHE_FILENAME)


def build_sic_map(quarters: Optional[List[str]] = None) -> Dict[str, int]:
    """Download the quarterly datasets and merge their cik→sic mappings."""
    quarters = quarters or recent_quarters()
    merged: Dict[str, int] = {}

    # Newest first, and earlier quarters must not overwrite a newer SIC —
    # companies do reclassify.
    for quarter in quarters:
        logging.info(f"SIC: fetching {quarter} dataset (~130 MB)")
        payload = sec_get(_DERA_URL.format(quarter=quarter), timeout=600)
        if payload is None:
            logging.warning(f"SIC: {quarter} unavailable, skipping")
            continue
        quarter_map = _parse_sub_file(payload)
        for cik, sic in quarter_map.items():
            merged.setdefault(cik, sic)
        logging.info(
            f"SIC: {quarter} contributed {len(quarter_map)} filers "
            f"({len(merged)} cumulative)"
        )

    if merged:
        try:
            with open(_cache_path(), "w") as handle:
                json.dump(merged, handle)
        except OSError as exc:
            logging.warning(f"SIC: could not write cache: {exc}")

    return merged


def get_sic_map(force_refresh: bool = False) -> Dict[str, int]:
    """The cached cik→sic map, built on first use."""
    if not force_refresh:
        path = _cache_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as handle:
                    cached = json.load(handle)
                if cached:
                    logging.info(f"SIC: {len(cached)} codes from cache")
                    return {str(k).zfill(10): int(v) for k, v in cached.items()}
            except (OSError, ValueError) as exc:
                logging.warning(f"SIC: ignoring unreadable cache: {exc}")

    return build_sic_map()


def get_model_map(force_refresh: bool = False) -> Dict[str, str]:
    """cik → valuation model name, for the whole known filer set."""
    return {cik: model_for_sic(sic) for cik, sic in get_sic_map(force_refresh).items()}
