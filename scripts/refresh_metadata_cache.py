#!/usr/bin/env python3
"""
Audit and refresh the per-symbol metadata cache.

The metadata cache stamps each entry with a `schema_version`. When required
fields are added (e.g. `country` in v3), the version is bumped and older
entries are treated as stale. This script reports stale entries and can delete
them so they are re-fetched on the next portfolio load.

Usage:
    python scripts/refresh_metadata_cache.py             # audit only
    python scripts/refresh_metadata_cache.py --clear     # delete stale entries
    python scripts/refresh_metadata_cache.py --dry-run --clear   # preview deletes
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config


V3_REQUIRED_KEYS = ("exchange", "country", "sector", "industry", "quoteType")


def _enrich_with_fmp(cache_dir: str, files: list, current_version: int, dry_run: bool) -> int:
    """Walk cache entries with missing country/sector/industry and fill them from FMP."""
    import time

    api_key = getattr(config, "FMP_API_KEY", None)
    if not api_key:
        print("FMP_API_KEY not set in .env — cannot enrich.")
        return 1

    candidates = []
    for fname in files:
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not entry.get("country") or not entry.get("sector") or not entry.get("industry"):
            candidates.append((fname, path, entry))

    if not candidates:
        print("\nNo entries need enrichment — country/sector/industry are populated everywhere.")
        return 0

    if dry_run:
        print(f"\nDRY RUN — {len(candidates)} entries are missing country/sector/industry.")
        sample = [c[0][:-5] for c in candidates[:20]]
        print(f"First 20: {', '.join(sample)}")
        print("Re-run without --dry-run to fetch from FMP and patch them in place.")
        return 0

    from fmp_provider import get_company_profile

    print(f"\nEnriching {len(candidates)} entries via FMP (~{len(candidates) * 0.3:.0f}s with throttle)...")
    enriched = 0
    failed = 0
    for i, (fname, path, entry) in enumerate(candidates, 1):
        symbol = fname[:-5]  # strip .json
        profile = get_company_profile(symbol, api_key)
        if i % 25 == 0:
            print(f"  [{i}/{len(candidates)}] processed...")
        if not profile:
            failed += 1
            continue

        filled = []
        for key in ("country", "sector", "industry", "currency", "exchange", "fullExchangeName", "quoteType", "name"):
            if not entry.get(key) and profile.get(key):
                entry[key] = profile[key]
                filled.append(key)

        if filled:
            entry["enriched_by"] = "fmp"
            entry["schema_version"] = current_version
            try:
                with open(path, "w") as f:
                    json.dump(entry, f, indent=2)
            except OSError as e:
                print(f"  {symbol}: write failed: {e}")
                continue
            enriched += 1
        # Throttle so we stay under FMP free-tier rate limit (~300/min)
        time.sleep(0.25)

    print(f"\nEnriched {enriched}/{len(candidates)} entries"
          + (f" ({failed} FMP lookups failed)" if failed else ""))
    return 0


def classify(entry: dict, current_version: int) -> str:
    version = entry.get("schema_version", 0)
    if version >= current_version:
        return "current"
    # Unstamped but has all required keys — grandfathered as current by the loader.
    if all(k in entry for k in V3_REQUIRED_KEYS):
        return "grandfathered"
    if version == 0:
        return "missing_fields"
    return f"stale_v{version}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true",
                        help="Delete stale entries (below current schema version).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without removing anything.")
    parser.add_argument("--enrich-with-fmp", action="store_true",
                        help="For entries missing country/sector/industry, fetch from FMP and patch in place. "
                             "Requires FMP_API_KEY in .env.")
    args = parser.parse_args()

    cache_dir = os.path.join(config.get_app_data_dir(), "cache", "metadata_cache")
    if not os.path.isdir(cache_dir):
        print(f"Metadata cache directory not found: {cache_dir}")
        return 1

    files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    current_version = config.METADATA_SCHEMA_VERSION

    print(f"Metadata cache: {cache_dir}")
    print(f"Current schema version: v{current_version}")
    print(f"Found {len(files)} cache entries\n")

    counts: Counter = Counter()
    stale_files: list[str] = []
    unreadable: list[str] = []

    for fname in files:
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            unreadable.append(fname)
            counts["unreadable"] += 1
            continue

        bucket = classify(entry, current_version)
        counts[bucket] += 1
        # Stale = anything the loader will reject. Grandfathered entries pass through.
        if bucket not in ("current", "grandfathered"):
            stale_files.append(fname)

    print("Status breakdown:")
    for bucket in sorted(counts):
        print(f"  {bucket:14s} {counts[bucket]:5d}")

    if args.enrich_with_fmp:
        return _enrich_with_fmp(cache_dir, files, current_version, dry_run=args.dry_run)

    if not args.clear:
        if stale_files:
            print(f"\n{len(stale_files)} stale entries would be invalidated.")
            print("Re-run with --clear to delete them (they will be re-fetched on next portfolio load).")
            print("Or run with --enrich-with-fmp to fill country/sector/industry from FMP without re-fetching everything.")
        return 0

    if not stale_files:
        print("\nNothing to clear — all entries are current.")
        return 0

    print(f"\n{'DRY RUN — would delete' if args.dry_run else 'Deleting'} "
          f"{len(stale_files)} stale entries...")
    deleted = 0
    for fname in stale_files:
        path = os.path.join(cache_dir, fname)
        if args.dry_run:
            continue
        try:
            os.remove(path)
            deleted += 1
        except OSError as e:
            print(f"  failed to delete {fname}: {e}")

    if args.dry_run:
        print("Dry run complete — no files removed.")
    else:
        print(f"Deleted {deleted}/{len(stale_files)} stale entries.")
        print("Next portfolio load will re-fetch fresh metadata from yfinance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
