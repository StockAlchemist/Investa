# -*- coding: utf-8 -*-
"""
Shared HTTP layer for SEC EDGAR requests.

The SEC requires every automated client to declare a contactable User-Agent and
to stay under 10 requests/second. Both rules are enforced here rather than at
each call site, so no caller can accidentally violate them: `sec_get` serialises
through a process-wide throttle.

Set INVESTA_SEC_USER_AGENT to override the declared contact string. The default
is derived from the repository owner and is adequate for personal research use,
but anything running on shared infrastructure should set it explicitly.

Public surface is deliberately small: `sec_get` (bytes) and `sec_get_json`.
Neither raises — failures are logged and return None, matching the
best-effort convention used by `fmp_provider`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

# SEC fair-access policy: 10 requests/second, declared User-Agent.
# We aim slightly below the limit to leave headroom for clock jitter.
_MAX_REQUESTS_PER_SECOND = 8.0
_MIN_INTERVAL = 1.0 / _MAX_REQUESTS_PER_SECOND

_DEFAULT_USER_AGENT = "Investa/1.0 (kittiwit@gmail.com)"

_throttle_lock = threading.Lock()
_last_request_ts = 0.0


def get_user_agent() -> str:
    """The contact string sent to the SEC on every request."""
    return (
        os.environ.get("INVESTA_SEC_USER_AGENT", _DEFAULT_USER_AGENT).strip()
        or _DEFAULT_USER_AGENT
    )


def _wait_for_slot() -> None:
    """Block until the next request is allowed under the rate limit."""
    global _last_request_ts
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_ts
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_request_ts = time.monotonic()


def sec_get(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    """
    Fetch a URL from sec.gov, respecting the rate limit. Returns None on failure.

    Retries on 429/503 (the SEC's throttling responses) with linear backoff.
    A 404 is treated as a definitive answer and is not retried — many companies
    legitimately have no data for a given endpoint.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": get_user_agent(),
            "Accept-Encoding": "gzip",
        },
    )

    for attempt in range(retries):
        _wait_for_slot()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip

                    payload = gzip.decompress(payload)
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logging.debug(f"EDGAR: 404 for {url}")
                return None
            if exc.code in (429, 503) and attempt < retries - 1:
                backoff = 2.0 * (attempt + 1)
                logging.warning(f"EDGAR: {exc.code} for {url}, backing off {backoff}s")
                time.sleep(backoff)
                continue
            logging.error(f"EDGAR: HTTP {exc.code} for {url}")
            return None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
            logging.error(f"EDGAR: request failed for {url}: {exc}")
            return None
    return None


def sec_get_json(url: str, timeout: int = 60) -> Optional[Any]:
    """Fetch and parse a JSON document from sec.gov. Returns None on failure."""
    payload = sec_get(url, timeout=timeout)
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logging.error(f"EDGAR: malformed JSON from {url}: {exc}")
        return None
