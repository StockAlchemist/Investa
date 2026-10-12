#!/usr/bin/env python3
"""Benchmark real analytics routes on identical deterministic, cached history.

Run each revision in a separate process with --source /path/to/src. No app
lifespan, user database, authentication credentials, or market network is used.
The history dependency alone is replaced; risk/backtest math runs unchanged.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def distribution(values):
    ordered = sorted(values)
    return {
        "median_ms": round(statistics.median(ordered), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "samples": [round(value, 3) for value in values],
    }


async def benchmark(samples):
    import httpx
    import numpy as np
    import pandas as pd
    from fastapi import FastAPI
    from server.dependencies import get_transaction_data
    from server.routes import analytics

    rng = np.random.default_rng(20261012)
    index = pd.bdate_range(end="2026-10-09", periods=5040)
    wealth = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, len(index))))
    benchmark_wealth = 100 * np.exp(np.cumsum(rng.normal(0.00025, 0.008, len(index))))
    history = pd.DataFrame(
        {
            "Portfolio Accumulated Gain": wealth,
            "^GSPC Price": benchmark_wealth,
            "Portfolio Value": wealth * 1000,
        },
        index=index,
    )
    data = (
        pd.DataFrame({"Date": [index[0]]}),
        {},
        {},
        set(),
        {},
        {},
        "/benchmark.db",
        0,
    )

    async def cached_history(**kwargs):
        return history.copy(), {}, {}, "benchmark: cached history"

    analytics._get_historical_performance_cached = cached_history
    app = FastAPI()
    app.include_router(analytics.router, prefix="/api")

    async def transaction_data():
        return data

    app.dependency_overrides[get_transaction_data] = transaction_data

    @app.get("/benchmark/probe")
    async def probe():
        return {"ok": True}

    result = {
        "history_points": len(history),
        "samples": samples,
        "python": platform.python_version(),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://benchmark"
    ) as client:
        for endpoint in ("projection/backtest", "risk_metrics"):
            path = f"/api/{endpoint}?currency=USD"
            warmup = await client.get(path)
            warmup.raise_for_status()
            body = warmup.json()
            if endpoint == "projection/backtest":
                assert body.get("available"), body
            assert "error" not in body, body
            expected_digest = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()
            ).hexdigest()
            durations, probe_delays, timer_lags = [], [], []
            for _ in range(samples):
                start = time.perf_counter()
                target = start + 0.005

                async def timed_analytics():
                    response = await client.get(path)
                    response.raise_for_status()
                    elapsed = (time.perf_counter() - start) * 1000
                    digest = hashlib.sha256(
                        json.dumps(response.json(), sort_keys=True).encode()
                    ).hexdigest()
                    assert digest == expected_digest, (
                        "Analytics output changed between repetitions"
                    )
                    return elapsed

                async def scheduled_probe():
                    await asyncio.sleep(max(0, target - time.perf_counter()))
                    timer_lag = max(0, (time.perf_counter() - target) * 1000)
                    response = await client.get("/benchmark/probe")
                    response.raise_for_status()
                    return max(0, (time.perf_counter() - target) * 1000), timer_lag

                analytics_time, (probe_delay, timer_lag) = await asyncio.gather(
                    timed_analytics(), scheduled_probe()
                )
                durations.append(analytics_time)
                probe_delays.append(probe_delay)
                timer_lags.append(timer_lag)
            result[endpoint] = {
                "analytics": distribution(durations),
                "probe_completion_delay": distribution(probe_delays),
                "event_loop_timer_lag": distribution(timer_lags),
                "response_sha256": expected_digest,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("Use at least three samples")
    sys.path.insert(0, str(args.source.resolve()))
    # Disposable source snapshots contain no .env. Avoid generating a signing key.
    os.environ["AUTH_SECRET_KEY"] = "isolated-benchmark-only"
    result = asyncio.run(benchmark(args.samples))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
