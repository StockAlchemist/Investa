# -*- coding: utf-8 -*-
"""
Batch worker for the Buffett/value ranking.

Runs independently of the API process, like the other workers in this
directory, because a full run over ~5,500 filers takes minutes to tens of
minutes and must never sit inside a request.

Refresh strategy: fundamentals move once a quarter, prices move continuously.
So a full run — which re-reads EDGAR and recomputes every quality pillar — is
expensive and rarely needed, while the value half goes stale within a day. The
worker defaults to a daily cadence, which is the useful compromise: quality
scores are recomputed from a local database (cheap), and the market data that
actually changed is refetched.

Usage:

    python src/buffett_rank_worker.py                # one run, then exit
    python src/buffett_rank_worker.py --loop         # run daily
    python src/buffett_rank_worker.py --limit 500    # smoke test, NOT saved
    python src/buffett_rank_worker.py --refresh-edgar  # re-ingest filings first

Note that `--limit` does not write to the store. Every strategy reads the newest
snapshot, so a capped run that got saved would quietly become the ranking the
app trades off — see `run_once`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import buffett_pipeline  # noqa: E402

DEFAULT_INTERVAL_SECONDS = 24 * 3600

# Retry pause after a failed run. Long enough not to hammer a broken upstream,
# short enough that a transient SEC or yfinance outage does not cost a day.
_ERROR_BACKOFF_SECONDS = 1800


def refresh_edgar() -> None:
    """
    Re-download and re-ingest SEC filings.

    Only worth doing weekly at most: companies file annually, and the bulk
    archive is ~1.4 GB.
    """
    import edgar_provider
    import universe

    logging.info("Worker: refreshing EDGAR bulk data")
    archive = edgar_provider.download_bulk_archive(force=True)
    if not archive:
        logging.error("Worker: EDGAR refresh failed, continuing with existing data")
        return

    ciks = set(universe.group_by_cik(universe.get_rankable_universe(force_refresh=True)))
    stats = edgar_provider.ingest_bulk(archive, ciks=ciks)
    logging.info(f"Worker: EDGAR refresh complete — {stats}")


def run_once(
    limit=None, skip_market_data: bool = False, persist_partial: bool = False
) -> bool:
    """
    Execute one ranking run. Returns True on success.

    **A capped run is not persisted unless asked for.** `--limit` exists to
    smoke-test the pipeline, but a completed run is a completed run as far as
    the store is concerned: it becomes the newest snapshot, and every strategy
    reads the newest snapshot. That has already gone wrong once — a five-filer
    smoke test left the Strategies tab serving a two-name book against a
    twenty-name rule. Computing without saving keeps the smoke test useful and
    harmless; `--persist-partial` is there for the rare case where a capped run
    really is meant to be the production ranking.
    """
    persist = limit is None or persist_partial
    if limit is not None and not persist:
        logging.warning(
            f"Worker: --limit {limit} is a smoke test, so this run will NOT be saved "
            "(it would otherwise become the ranking every strategy reads). "
            "Pass --persist-partial to override."
        )
    try:
        result = buffett_pipeline.run(
            limit=limit, persist=persist, skip_market_data=skip_market_data
        )
        logging.info(
            f"Worker: run {result.run_id} finished in {result.duration_seconds:.0f}s "
            f"— {result.stats}"
        )
        return True
    except Exception as exc:
        logging.error(f"Worker: ranking run failed: {exc}", exc_info=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="Run repeatedly")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between runs in loop mode (default: daily)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of filers. A smoke test: the run is computed but NOT saved, "
             "because a capped run would otherwise become the snapshot every strategy reads.",
    )
    parser.add_argument(
        "--persist-partial",
        action="store_true",
        help="Save a --limit run anyway. Only for when a capped run really is meant to "
             "become the production ranking.",
    )
    parser.add_argument(
        "--skip-market-data",
        action="store_true",
        help="Quality-only run with no network calls",
    )
    parser.add_argument(
        "--refresh-edgar",
        action="store_true",
        help="Re-download and re-ingest SEC filings before ranking",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.refresh_edgar:
        refresh_edgar()

    if not args.loop:
        return 0 if run_once(args.limit, args.skip_market_data, args.persist_partial) else 1

    logging.info(f"Worker: entering loop, interval {args.interval}s")
    while True:
        succeeded = run_once(args.limit, args.skip_market_data, args.persist_partial)
        delay = args.interval if succeeded else _ERROR_BACKOFF_SECONDS
        logging.info(f"Worker: sleeping {delay}s")
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(main())
