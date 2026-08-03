"""Turn the latest stored ranking into a sized buy list.

Reads the most recent run from `buffett_ranks.db`, re-blends quality and value at
whatever weight is asked for, optionally caps how many names may come from one
industry, and prints an equal-weight order list for a given amount of capital.

Two deliberate choices:

**Re-blending rather than re-running.** The stored run keeps `quality_score` and
`value_score` separately, so changing the blend is arithmetic on data already
computed — no EDGAR reads, no quotes, and no risk of the list drifting from the
ranking the app itself shows.

**Industry caps use SIC, applied greedily down the ranked list.** The ranking
scores companies one at a time and has no concept of correlation, so it will
put four mortgage insurers in a twenty-name book. Capping skips a name once its
industry is full and fills the slot from further down, which costs a little
score and removes a lot of shared risk. Note what SIC cannot see: mortgage
insurers (6351) and homebuilders (1531) are different industries carrying the
same housing exposure, and no SIC cap will group them.

    python scripts/buffett_live_picks.py --capital 200000 --quality 0.8 \
        --max-per-sector 3 --sector-digits 2
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
import edgar_sic  # noqa: E402

# Major-group names for the industries this ranking actually surfaces. Only used
# to make the output readable — the cap works on the numeric code.
_SIC_GROUPS = {
    "15": "Homebuilders / construction",
    "20": "Food products",
    "23": "Apparel",
    "28": "Pharma / chemicals",
    "30": "Rubber & plastics",
    "34": "Fabricated metal",
    "35": "Industrial machinery",
    "36": "Electronics",
    "37": "Transport equipment",
    "38": "Instruments",
    "45": "Air transport",
    "48": "Communications",
    "50": "Wholesale (durables)",
    "51": "Wholesale (nondurables)",
    "52": "Building materials retail",
    "53": "General merchandise",
    "54": "Food retail",
    "56": "Apparel retail",
    "57": "Furniture retail",
    "58": "Restaurants",
    "59": "Miscellaneous retail",
    "60": "Banks",
    "61": "Non-bank credit",
    "62": "Brokers & exchanges",
    "63": "Insurance carriers",
    "64": "Insurance agents",
    "65": "Real estate",
    "67": "Holding & investment",
    "70": "Hotels",
    "73": "Business services / software",
    "79": "Recreation",
    "80": "Health services",
    "87": "Consulting & engineering",
}

# Codes whose major group is misleading on its own: 6798 is a REIT, not a
# generic holding company, and 6351 is mortgage insurance rather than insurance
# at large — which is exactly the distinction that matters for concentration.
_SIC_EXACT = {
    "6798": "REITs",
    "6351": "Mortgage insurance",
    "1531": "Homebuilders",
    "5500": "Auto dealers & services",
}


def _industry_label(sic: str) -> str:
    if not sic:
        return "Unclassified"
    return _SIC_EXACT.get(sic) or _SIC_GROUPS.get(sic[:2], "Other")


def _db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "buffett_ranks.db")


def load_latest_run(run_id: Optional[int] = None) -> tuple:
    with sqlite3.connect(_db_path()) as conn:
        if run_id is None:
            row = conn.execute(
                "SELECT run_id, finished_at, ranked_count FROM rank_runs "
                "WHERE finished_at IS NOT NULL ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise SystemExit(
                    "No completed ranking run found — run the Buffett pipeline first"
                )
            run_id, finished_at, ranked = row
        else:
            finished_at, ranked = conn.execute(
                "SELECT finished_at, ranked_count FROM rank_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        frame = pd.read_sql(
            "SELECT * FROM rank_scores WHERE run_id = ? ", conn, params=(run_id,)
        )
    return run_id, finished_at, ranked, frame


def blend(frame: pd.DataFrame, quality_weight: float) -> pd.Series:
    quality = pd.to_numeric(frame["quality_score"], errors="coerce")
    value = pd.to_numeric(frame["value_score"], errors="coerce")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(1.0)
    blended = quality.where(
        value.isna(), quality * quality_weight + value * (1.0 - quality_weight)
    )
    return blended * confidence


def apply_cap(
    frame: pd.DataFrame,
    scores: pd.Series,
    top_n: int,
    max_per_sector: Optional[int],
    digits: int,
) -> pd.DataFrame:
    order = scores.sort_values(ascending=False).index
    sic_map = edgar_sic.get_sic_map()

    sectors: Dict[int, str] = {}
    for index, cik in frame["cik"].items():
        try:
            key = str(int(float(cik))).zfill(10)
        except (TypeError, ValueError):
            continue
        sic = sic_map.get(key)
        if sic is not None:
            sectors[index] = str(int(sic)).zfill(4)

    counts: Dict[str, int] = {}
    picked: List[int] = []
    skipped: List[tuple] = []
    for index in order:
        full_sic = sectors.get(index, "")
        group = full_sic[:digits] if full_sic else f"?{index}"
        if max_per_sector and counts.get(group, 0) >= max_per_sector:
            skipped.append((frame.loc[index, "symbol"], full_sic, group))
            continue
        counts[group] = counts.get(group, 0) + 1
        picked.append(index)
        if len(picked) == top_n:
            break

    result = frame.loc[picked].copy()
    result["sic"] = [sectors.get(i, "") for i in picked]
    result["industry"] = [_industry_label(sectors.get(i, "")) for i in picked]
    result["score"] = scores.loc[picked]
    result.attrs["skipped"] = skipped
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--quality", type=float, default=0.8)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-per-sector", type=int, default=None)
    parser.add_argument("--sector-digits", type=int, default=2)
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args()

    run_id, finished_at, ranked, frame = load_latest_run(args.run_id)
    scores = blend(frame, args.quality)
    picks = apply_cap(frame, scores, args.top, args.max_per_sector, args.sector_digits)

    per_position = args.capital / args.top
    picks["shares"] = (per_position / picks["price"]).apply(
        lambda x: math.floor(x) if pd.notna(x) else 0
    )
    picks["cost"] = picks["shares"] * picks["price"]
    picks["weight"] = picks["cost"] / args.capital * 100

    cap_label = (
        f"max {args.max_per_sector} per {args.sector_digits}-digit SIC"
        if args.max_per_sector
        else "no industry cap"
    )
    print(f"Run {run_id} finished {finished_at} — {ranked} companies ranked")
    print(
        f"Blend: {args.quality:.0%} quality / {1 - args.quality:.0%} value · "
        f"top {args.top} equal weight · {cap_label}"
    )
    print(f"Capital: ${args.capital:,.0f} · target ${per_position:,.0f} per position\n")

    display = picks[
        ["symbol", "name", "industry", "price", "shares", "cost", "weight", "score"]
    ].copy()
    display["name"] = display["name"].str.slice(0, 26)
    print(
        display.to_string(
            index=False,
            formatters={
                "price": lambda x: f"{x:,.2f}",
                "cost": lambda x: f"{x:,.0f}",
                "weight": lambda x: f"{x:.2f}%",
                "score": lambda x: f"{x:.1f}",
            },
        )
    )

    invested = picks["cost"].sum()
    print(f"\nInvested ${invested:,.0f} · cash left ${args.capital - invested:,.0f}")

    print("\nIndustry concentration")
    concentration = (
        picks.groupby("industry")["cost"].sum().sort_values(ascending=False)
        / args.capital
        * 100
    )
    for industry, weight in concentration.items():
        count = int((picks["industry"] == industry).sum())
        print(f"  {weight:5.1f}%  {industry} ({count})")

    skipped = picks.attrs.get("skipped") or []
    if skipped:
        print(f"\nSkipped by the cap ({len(skipped)}), in rank order:")
        for symbol, sic, group in skipped[:15]:
            print(f"  {symbol:6s} SIC {sic} (group {group} already full)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
