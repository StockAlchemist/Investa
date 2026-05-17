# -*- coding: utf-8 -*-
"""
Background asyncio worker that walks the per-symbol metadata cache and
re-fetches stale entries on a schedule.

A "stale" entry is one whose `schema_version` is below the current
METADATA_SCHEMA_VERSION AND doesn't have all the legacy required keys.
This mirrors the validation logic in `MarketDataProvider._ensure_metadata_batch`
so the worker only refreshes what the loader would actually reject.

Why a worker at all? Without one, stale entries linger until the user happens
to load a portfolio that touches them. With it, the cache silently catches up
in the background even when nobody's looking at the affected symbols.

Tuning knobs (env):
    INVESTA_METADATA_REFRESH_INTERVAL — seconds between cycles (default 6h)
    INVESTA_METADATA_REFRESH_BATCH    — max symbols per cycle (default 50)
    INVESTA_METADATA_REFRESH_ENABLED  — "0" to disable entirely
"""
import asyncio
import json
import logging
import os
from typing import List

import config

logger = logging.getLogger(__name__)


REFRESH_INTERVAL_SECONDS = int(os.getenv("INVESTA_METADATA_REFRESH_INTERVAL", str(6 * 3600)))
BATCH_SIZE = int(os.getenv("INVESTA_METADATA_REFRESH_BATCH", "50"))
ENABLED = os.getenv("INVESTA_METADATA_REFRESH_ENABLED", "1") != "0"

_V3_REQUIRED_KEYS = ("exchange", "country", "sector", "industry", "quoteType")


def _find_stale_symbols(cache_dir: str, current_version: int, limit: int) -> List[str]:
    """Scan the metadata cache and return up to `limit` symbols that need refresh."""
    if not os.path.isdir(cache_dir):
        return []

    stale: List[str] = []
    try:
        files = os.listdir(cache_dir)
    except OSError as e:
        logger.warning(f"Cannot list metadata cache dir {cache_dir}: {e}")
        return []

    for fname in files:
        if not fname.endswith(".json"):
            continue
        if len(stale) >= limit:
            break

        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Skip entries that are already at the current version.
        if entry.get("schema_version", 0) >= current_version:
            continue
        # Skip grandfathered entries — same compat rule as _ensure_metadata_batch.
        if all(k in entry for k in _V3_REQUIRED_KEYS):
            continue

        stale.append(fname[:-5])  # strip .json
    return stale


def _refresh_batch_sync(symbols: List[str]) -> int:
    """
    Synchronous refresh: invokes the existing batch fetcher (which includes the
    FMP enrichment fallback from #2). Returns the number of entries actually
    written. Designed to be called from `asyncio.to_thread`.
    """
    from market_data import get_shared_mdp  # local import — keeps cold-start fast

    if not symbols:
        return 0

    try:
        mdp = get_shared_mdp()
    except Exception as e:
        logger.warning(f"Cannot acquire shared MarketDataProvider: {e}")
        return 0

    # Force re-fetch by deleting the stale files first; the batch fetcher
    # writes fresh entries (with current schema_version) afterwards.
    cache_dir = os.path.join(config.get_app_data_dir(), "cache", "metadata_cache")
    deleted = 0
    for sym in symbols:
        path = os.path.join(cache_dir, sym + ".json")
        try:
            os.remove(path)
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.debug(f"Could not delete stale entry {sym}: {e}")

    # Now re-populate. This is the slow call (subprocess to yfinance) — that's
    # why we run it in a thread.
    try:
        refreshed = mdp._ensure_metadata_batch(set(symbols))
        return len(refreshed)
    except Exception as e:
        logger.warning(f"Batch metadata refresh failed: {e}")
        return 0


async def refresh_loop() -> None:
    """Periodic refresh loop. Cancellable; logs each cycle's outcome."""
    if not ENABLED:
        logger.info("Metadata refresh worker disabled by INVESTA_METADATA_REFRESH_ENABLED=0")
        return

    cache_dir = os.path.join(config.get_app_data_dir(), "cache", "metadata_cache")
    current_version = config.METADATA_SCHEMA_VERSION

    logger.info(
        f"Metadata refresh worker started — interval={REFRESH_INTERVAL_SECONDS}s, "
        f"batch={BATCH_SIZE}, target_version=v{current_version}"
    )

    # Delay first run so we don't compete with cold-start work.
    await asyncio.sleep(120)

    while True:
        try:
            symbols = _find_stale_symbols(cache_dir, current_version, BATCH_SIZE)
            if symbols:
                logger.info(f"Metadata refresh: refreshing {len(symbols)} stale entries")
                refreshed = await asyncio.to_thread(_refresh_batch_sync, symbols)
                logger.info(f"Metadata refresh: completed ({refreshed} entries written)")
            else:
                logger.debug("Metadata refresh: no stale entries this cycle")
        except asyncio.CancelledError:
            logger.info("Metadata refresh worker cancelled")
            raise
        except Exception as e:
            logger.exception(f"Metadata refresh cycle errored (will retry next interval): {e}")

        try:
            await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("Metadata refresh worker cancelled during sleep")
            raise
