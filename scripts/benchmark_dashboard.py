#!/usr/bin/env python3
"""Measure a running backend without printing portfolio data or credentials.

INVESTA_BENCHMARK_TOKEN=<token> python scripts/benchmark_dashboard.py --currency USD
The first request is reported separately; it is cold only if server caches are cold.
No cache clearing, market refresh, or portfolio writes are performed.
"""

import argparse
import json
import os
import statistics
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def measure(base: str, endpoint: str, query: dict, samples: int, token: str):
    durations, sizes = [], []
    for _ in range(samples):
        request = Request(
            f"{base.rstrip('/')}/{endpoint}?{urlencode(query)}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
        start = time.perf_counter()
        with urlopen(request, timeout=120) as response:
            body = response.read()
        durations.append((time.perf_counter() - start) * 1000)
        sizes.append(len(body))
    warm = sorted(durations[1:])
    return {
        "first_request_ms": round(durations[0], 2),
        "warm_median_ms": round(statistics.median(warm), 2),
        "warm_p95_ms": round(warm[min(len(warm) - 1, int(len(warm) * 0.95))], 2),
        "response_bytes_max": max(sizes),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000/api")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--samples", type=int, default=11)
    args = parser.parse_args()
    token = os.environ.get("INVESTA_BENCHMARK_TOKEN")
    if not token:
        parser.error("Set INVESTA_BENCHMARK_TOKEN to an authenticated bearer token")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    result = {}
    for endpoint, extra in [
        ("summary/headline", {}),
        ("summary", {}),
        ("holdings", {}),
        ("history", {"period": "1y", "interval": "1d"}),
    ]:
        result[endpoint] = measure(
            args.base,
            endpoint,
            {"currency": args.currency, **extra},
            args.samples,
            token,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
