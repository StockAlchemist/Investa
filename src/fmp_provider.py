# -*- coding: utf-8 -*-
"""
Financial Modeling Prep (FMP) provider.

Used as a fallback when yfinance returns incomplete metadata — most notably
the `country` field, which yfinance omits for many older cache entries and
some thinly-covered tickers. FMP correctly resolves country-of-origin even
for ADRs (e.g. ASML → Netherlands, TSM → Taiwan).

Public surface is intentionally narrow: a single `get_company_profile(symbol)`
that returns a dict normalised to the same field names yfinance uses, or
None on any failure. Errors are logged but never raised — callers should
treat FMP as a best-effort enrichment layer.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

# ISO Alpha-2 → English country name. Limited to the codes we've seen in
# real portfolios; FMP may return others, in which case we keep the code.
_COUNTRY_CODES = {
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    "GB": "United Kingdom",
    "IE": "Ireland",
    "NL": "Netherlands",
    "DE": "Germany",
    "FR": "France",
    "CH": "Switzerland",
    "SE": "Sweden",
    "DK": "Denmark",
    "NO": "Norway",
    "FI": "Finland",
    "BE": "Belgium",
    "IT": "Italy",
    "ES": "Spain",
    "PT": "Portugal",
    "LU": "Luxembourg",
    "AT": "Austria",
    "JP": "Japan",
    "CN": "China",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "KR": "South Korea",
    "SG": "Singapore",
    "TH": "Thailand",
    "IN": "India",
    "ID": "Indonesia",
    "VN": "Vietnam",
    "MY": "Malaysia",
    "PH": "Philippines",
    "AU": "Australia",
    "NZ": "New Zealand",
    "BR": "Brazil",
    "AR": "Argentina",
    "CL": "Chile",
    "ZA": "South Africa",
    "IL": "Israel",
    "AE": "United Arab Emirates",
    "SA": "Saudi Arabia",
    "TR": "Turkey",
    "RU": "Russia",
    "BM": "Bermuda",
    "KY": "Cayman Islands",
}

_BASE_URL = "https://financialmodelingprep.com/stable/profile"
_DEFAULT_TIMEOUT = 8


def _normalise_country(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return _COUNTRY_CODES.get(code.upper(), code)


def get_company_profile(symbol: str, api_key: str, timeout: int = _DEFAULT_TIMEOUT) -> Optional[dict]:
    """
    Fetch FMP company profile for `symbol`. Returns a dict with the same keys
    yfinance uses (sector, industry, country, exchange, currency, quoteType,
    name) so it can be drop-in merged into a metadata cache entry, or None on
    any error (invalid key, network failure, rate limit, missing symbol).
    """
    if not symbol or not api_key:
        return None

    url = f"{_BASE_URL}?symbol={symbol}&apikey={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        logging.warning(f"FMP profile fetch failed for {symbol}: HTTP {e.code}")
        return None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        logging.warning(f"FMP profile fetch failed for {symbol}: {e}")
        return None

    if not payload or not isinstance(payload, list):
        return None

    profile = payload[0]
    if not isinstance(profile, dict):
        return None

    # FMP uses `isEtf`/`isFund` flags rather than yfinance's quoteType string.
    quote_type = "EQUITY"
    if profile.get("isEtf"):
        quote_type = "ETF"
    elif profile.get("isFund"):
        quote_type = "MUTUALFUND"

    return {
        "name": profile.get("companyName") or symbol,
        "currency": profile.get("currency"),
        "sector": profile.get("sector") or None,
        "industry": profile.get("industry") or None,
        "country": _normalise_country(profile.get("country")),
        "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
        "fullExchangeName": profile.get("exchange"),
        "quoteType": quote_type,
    }
