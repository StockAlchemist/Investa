"""Strategy routes, plus the advisory market-trend indicator.

`/trend-signal` is the one the dashboard polls, so it is cached separately and
kept cheap: it reads one symbol's daily closes and reduces them to a dozen
numbers. It sits outside `/strategies/` because no strategy acts on it — it is
market context, not a rule. The allocation endpoint is heavier (it re-blends
the whole ranking snapshot) and is not on the dashboard's critical path.

Nothing here places an order or mutates portfolio state. These endpoints answer
"what does the rule say today"; acting on it stays a deliberate, manual step.
"""

# ruff: noqa: E402
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

import strategies as strategy_lib
from server.auth import User
from server.dependencies import get_current_user
from server.route_utils import SWRCache, clean_nans

router = APIRouter()

# The signal moves once a day at most (it is built from daily closes), but the
# dashboard mounts on every navigation. Stale-while-revalidate keeps the card
# instant while a background refresh picks up the day's close.
_SIGNAL_CACHE = SWRCache()
_SIGNAL_TTL_SECONDS = 900


@router.get("/strategies")
async def list_strategies(current_user: User = Depends(get_current_user)):
    """
    The strategy catalogue: rules, parameters, backtested record and risks.

    Static data — no market access — so a client can render the picker before
    any price request resolves.
    """
    payload = [strategy_lib.strategy_payload(s) for s in strategy_lib.list_strategies()]
    return clean_nans(
        {"strategies": payload, "default": strategy_lib.DEFAULT_STRATEGY_ID}
    )


@router.get("/trend-signal")
async def get_trend_signal(
    symbol: str = Query(
        strategy_lib.MARKET_SIGNAL_SYMBOL,
        description="Index or ETF to read the trend of",
    ),
    sma_months: int = Query(
        strategy_lib.MARKET_SIGNAL_SMA_MONTHS,
        ge=2,
        le=24,
        description="Moving-average length, in months",
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Market-trend indicator: one index's close against its own moving average.

    **Advisory only.** No strategy acts on this — gating a stock book with it
    was measured and rejected. It is market context, and the payload says so
    via `advisory_only`. Deliberately not under `/strategies/` for that reason.

    `state` is the *active* reading, set at the last completed month-end.
    `provisional_state` is what the comparison would say if the month ended
    today; it previews the next month's reading and is kept a separate field so
    a client cannot present a mid-month price as the current state.
    """
    cache_key = f"{symbol.upper()}:{sma_months}"

    async def compute():
        return await run_in_threadpool(
            strategy_lib.market_trend_signal, symbol.upper(), sma_months
        )

    try:
        signal = await _SIGNAL_CACHE.get_or_compute(
            cache_key, _SIGNAL_TTL_SECONDS, compute
        )
    except Exception as exc:
        logging.error(f"Strategies: signal computation failed: {exc}")
        raise HTTPException(
            status_code=503, detail="Could not compute the trend signal"
        )

    if signal is None:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough price history for {symbol.upper()} to form the signal",
        )
    return clean_nans(signal)


@router.get("/strategies/{strategy_id}/allocation")
async def get_allocation(
    strategy_id: str,
    capital: float = Query(..., gt=0, description="Amount to allocate, in USD"),
    current_user: User = Depends(get_current_user),
):
    """
    What the strategy says to hold today, sized to `capital`.

    Recomputed per request from the newest *completed* ranking run, so a fresh
    run is picked up immediately and a run still being written is never read.
    Deliberately uncached for that reason: which twenty companies to hold is
    the answer people act on, and it is cheap to derive.

    Which names to hold comes from the snapshot; the prices used to turn
    weights into share counts are live quotes, falling back to the snapshot's
    stored close when a quote is unavailable (`price_source` says which). The
    weights are the part of the answer that is actually fixed.
    """
    strategy = strategy_lib.get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{strategy_id}'")

    try:
        allocation = await run_in_threadpool(
            strategy_lib.build_allocation, strategy, capital
        )
    except Exception as exc:
        logging.error(f"Strategies: allocation failed for {strategy_id}: {exc}")
        raise HTTPException(status_code=500, detail="Could not build the allocation")

    return clean_nans(allocation)
