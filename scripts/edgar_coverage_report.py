"""Measure EDGAR concept coverage across the ranking universe.

This is the gate on the whole Buffett/value ranking build. The ranking assumes
it can compute returns on capital, leverage, cash generation and growth over a
long window for most of the market. That assumption is only worth holding if the
underlying XBRL data is actually there — and XBRL tag usage varies by company,
by year, and above all by industry.

The report answers three questions:

  1. **Presence** — for each logical concept, what fraction of companies have it
     at all? A concept below ~80% cannot carry a gate, because failing it would
     mostly measure disclosure habits rather than business quality.
  2. **Depth** — how many annual periods does the median company have? The
     durability pillar needs ~10; anything less caps what consistency scoring
     can honestly claim.
  3. **Chain effectiveness** — which tag in each fallback chain actually
     answered, and how often. A chain whose later entries never fire is
     over-specified; one where the last entry carries most periods is a sign the
     ordering is wrong.

Run from the repo root:

    python scripts/edgar_coverage_report.py [--sample 500] [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import random
import statistics
import sys
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import edgar_provider  # noqa: E402
import edgar_sic  # noqa: E402
import universe  # noqa: E402
from edgar_concepts import (  # noqa: E402
    BANK_CONCEPTS,
    REIT_CONCEPTS,
    all_concepts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edgar-coverage")

# Concepts every generic (non-financial, non-REIT) company must have for the
# ranking to say anything at all. Reported separately because a gap here is
# disqualifying, whereas a gap in, say, `inventory` merely means the company
# holds no inventory.
CORE_CONCEPTS = [
    "revenue",
    "net_income",
    "equity",
    "total_assets",
    "operating_cash_flow",
]


def measure(ciks: List[str], models: Dict[str, str]) -> Dict:
    """
    Walk the fact store and tally presence, depth and tag provenance.

    Tallies are kept per valuation model as well as overall. Measuring a bank
    concept against the whole universe answers the wrong question — `deposits`
    appearing for 13% of all companies says only that ~13% of the market is
    banks. What matters is whether a bank has deposits.
    """
    chains = all_concepts()
    present: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    depths: Dict[str, Dict[str, List[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    tag_hits: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    totals: collections.Counter = collections.Counter()

    for index, cik in enumerate(ciks, start=1):
        provenance = edgar_provider.get_concept_provenance(cik)
        if not provenance:
            continue
        model = models.get(cik, "generic")
        totals[model] += 1
        totals["all"] += 1

        for concept in chains:
            periods = provenance.get(concept)
            if not periods:
                continue
            present[model][concept] += 1
            present["all"][concept] += 1
            depths[model][concept].append(len(periods))
            depths["all"][concept].append(len(periods))
            for tag in periods.values():
                tag_hits[concept][tag] += 1

        if index % 250 == 0:
            log.info("measured %d/%d companies", index, len(ciks))

    return {
        "companies_requested": len(ciks),
        "totals": dict(totals),
        "present": {m: dict(c) for m, c in present.items()},
        "depths": {m: {k: sorted(v) for k, v in d.items()} for m, d in depths.items()},
        "tag_hits": {k: dict(v) for k, v in tag_hits.items()},
    }


def _pct(count: int, total: int) -> float:
    return (100.0 * count / total) if total else 0.0


def render(results: Dict) -> None:
    totals = results["totals"]
    total = totals.get("all", 0)
    chains = all_concepts()
    sector_concepts = set(BANK_CONCEPTS) | set(REIT_CONCEPTS)

    print()
    print("=" * 78)
    print("EDGAR CONCEPT COVERAGE")
    print("=" * 78)
    print(
        f"Companies sampled: {results['companies_requested']}   "
        f"with any EDGAR data: {total} "
        f"({_pct(total, results['companies_requested']):.1f}%)"
    )
    print(
        f"By model — generic: {totals.get('generic', 0)}, "
        f"bank: {totals.get('bank', 0)}, reit: {totals.get('reit', 0)}"
    )

    def section(title: str, concepts: List[str], model: str = "all") -> None:
        denominator = totals.get(model, 0)
        print()
        print(f"--- {title}  (n={denominator}) ---")
        if not denominator:
            print("  no companies of this type in the sample")
            return
        print(f"{'concept':32s} {'present':>9s} {'median yrs':>11s} {'>=10 yrs':>9s}")
        for concept in concepts:
            count = results["present"].get(model, {}).get(concept, 0)
            depth_list = results["depths"].get(model, {}).get(concept, [])
            median_depth = statistics.median(depth_list) if depth_list else 0
            deep = sum(1 for d in depth_list if d >= 10)
            print(
                f"{concept:32s} {_pct(count, denominator):8.1f}% {median_depth:11.0f} "
                f"{_pct(deep, denominator):8.1f}%"
            )

    section("Core (a gap here disqualifies a company)", CORE_CONCEPTS)
    generic = [c for c in chains if c not in sector_concepts and c not in CORE_CONCEPTS]
    section("Generic model", sorted(generic), model="generic")
    section("Banks & insurers — measured within banks only", sorted(BANK_CONCEPTS), model="bank")
    section("REITs — measured within REITs only", sorted(REIT_CONCEPTS), model="reit")

    print()
    print("--- Fallback chain effectiveness ---")
    print("(a chain earning all its hits from one tag is over-specified;")
    print(" one where a late entry dominates is probably mis-ordered)")
    print()
    for concept, chain in sorted(chains.items()):
        hits = results["tag_hits"].get(concept, {})
        if not hits or len(chain) == 1:
            continue
        total_hits = sum(hits.values())
        used = [(tag, n) for tag, n in hits.items() if n]
        if len(used) < 2:
            continue
        parts = ", ".join(
            f"{tag}={_pct(n, total_hits):.0f}%"
            for tag, n in sorted(used, key=lambda kv: -kv[1])[:4]
        )
        print(f"  {concept:28s} {parts}")

    print()
    print("--- Verdict ---")
    core_present = results["present"].get("all", {})
    weak_core = [c for c in CORE_CONCEPTS if _pct(core_present.get(c, 0), total) < 80.0]
    if weak_core:
        print(f"  BLOCKED: core concepts below 80% coverage: {', '.join(weak_core)}")
    else:
        print("  Core concepts all clear 80% — the generic model can proceed.")

    all_depths = results["depths"].get("all", {})
    depth_ok = [
        c
        for c in CORE_CONCEPTS
        if all_depths.get(c) and statistics.median(all_depths[c]) >= 10
    ]
    print(
        f"  {len(depth_ok)}/{len(CORE_CONCEPTS)} core concepts reach a 10-year "
        "median depth (durability scoring needs this)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        help="Number of companies to measure (0 = the whole universe)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed")
    parser.add_argument("--json", type=str, help="Also write raw results to this path")
    args = parser.parse_args()

    entries = universe.get_rankable_universe()
    if not entries:
        log.error("Universe is empty — cannot measure coverage")
        return 1

    # Collapse share classes: BRK-A and BRK-B share one set of filings, so
    # measuring both would overstate coverage confidence.
    by_cik = universe.group_by_cik(entries)
    ciks = sorted(by_cik)
    log.info("Universe: %d listings across %d distinct filers", len(entries), len(ciks))

    if args.sample and args.sample < len(ciks):
        random.seed(args.seed)
        ciks = random.sample(ciks, args.sample)
        log.info("Sampling %d filers (seed %d)", len(ciks), args.seed)

    models = edgar_sic.get_model_map()
    unmapped = sum(1 for cik in ciks if cik not in models)
    if unmapped:
        log.warning(
            "%d/%d filers have no SIC code and default to the generic model",
            unmapped,
            len(ciks),
        )

    results = measure(ciks, models)
    render(results)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        log.info("Raw results written to %s", args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
