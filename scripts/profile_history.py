#!/usr/bin/env python3
"""
In-process profiler for the heavy /history compute path.

Profiles `_calculate_historical_performance_internal` — the same function the
GET /api/history endpoint calls — against a real user's transactions, under
cProfile. No sudo and no auth required (unlike py-spy on a live server).

Usage:
    cd src && python ../scripts/profile_history.py [username] [period]

Defaults: username=kitmatan, period=max
Outputs a .prof file under profiles/ and prints the top hotspots by total
(self) time and cumulative time.
"""
import asyncio
import cProfile
import os
import pstats
import sys
import time

# Run as if from src/ (mirrors how uvicorn launches).
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)
os.chdir(SRC)

from server.auth import User  # noqa: E402
from server.dependencies import get_transaction_data  # noqa: E402
from server.api import _calculate_historical_performance_internal  # noqa: E402

USERNAME = sys.argv[1] if len(sys.argv) > 1 else "kitmatan"
PERIOD = sys.argv[2] if len(sys.argv) > 2 else "max"
CURRENCY = "THB"


def build_user(username: str) -> User:
    return User(id=1, username=username, alias=None, is_active=True, created_at="")


async def run_once(data):
    return await _calculate_historical_performance_internal(
        currency=CURRENCY,
        period=PERIOD,
        accounts=None,        # all accounts
        benchmarks=[],        # skip benchmark fetches (network) to isolate compute
        data=data,
        return_df=True,
        interval="1d",
        force=True,           # bypass snapshot/result caches -> real compute
    )


def main():
    user = build_user(USERNAME)
    print(f"Loading transactions for '{USERNAME}'...")
    data = get_transaction_data(user)
    df = data[0]
    print(f"  transactions loaded: {len(df)} rows")
    if df.empty:
        print("  ERROR: no transactions — nothing to profile.")
        sys.exit(1)

    # Profile the FIRST (cold) call in this fresh process. In-process caches
    # (daily-holdings L1, FIFO) are empty, so this measures the real cache-miss
    # cost — the only path that is actually slow. Numba JIT loads from its
    # on-disk cache (cache=True), so compile cost here is small if the app has
    # run before; any large numba-compiler frames in the output = one-time JIT.
    print(f"Profiled COLD run (period={PERIOD})...")
    os.makedirs(os.path.join(SRC, "..", "profiles"), exist_ok=True)
    out = os.path.abspath(os.path.join(SRC, "..", "profiles", f"history_{USERNAME}_{PERIOD}.prof"))
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    asyncio.run(run_once(data))
    pr.disable()
    wall = time.perf_counter() - t0
    pr.dump_stats(out)
    print(f"  cold wall time: {wall:.2f}s")
    print(f"  saved: {out}\n")

    st = pstats.Stats(pr)
    print("=" * 70)
    print("TOP 25 BY SELF TIME (tottime) — where the CPU actually burns:")
    print("=" * 70)
    st.sort_stats("tottime").print_stats(25)
    print("=" * 70)
    print("TOP 20 BY CUMULATIVE TIME (cumtime) — includes called subroutines:")
    print("=" * 70)
    st.sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    main()
