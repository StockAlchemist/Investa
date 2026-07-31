"""Calibration harness for the intrinsic-value models in `financial_ratios.py`.

Runs `get_comprehensive_intrinsic_value` over every symbol in the local
fundamentals cache (info + all three annual statements) and reports the
cross-sectional distribution of margin of safety, per-model coverage, and the
dispersion between models. No network access — the cache is the fixture.

Why margin of safety is the yardstick: across a broad universe the median stock
is, by construction, priced where the market clears. A model whose median MOS
is -33% is not being "conservative", it is asserting that the typical listed
company is worth a third of its price — a claim it cannot cash. Calibration
does not prove the model predicts returns (see `rank_signal_lab.py` and the
memo on why the DCF left the ranking), but a model that fails calibration is
unusable for the thing this one is actually shown for: a per-stock fair-value
readout next to a price.

Usage:
    python scripts/intrinsic_value_lab.py                # current code
    python scripts/intrinsic_value_lab.py --limit 300    # quick pass
    python scripts/intrinsic_value_lab.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import financial_ratios as fr  # noqa: E402
import market_data  # noqa: E402

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "cache", "fundamentals_cache")
)

# Percentiles reported for every distribution. p50 is the headline: it is the
# one the "median company is worth a third of its price" failure shows up in.
PCTS = [5, 10, 25, 50, 75, 90, 95]


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_statement(symbol: str, kind: str) -> Optional[pd.DataFrame]:
    """
    The statement as production assembles it: the cached Yahoo frame with the
    SEC-filed history merged over it.

    Reading the cache alone would test a model on four annual periods while the
    app runs it on nineteen, which is not a calibration of anything shipped. The
    merge is a local SQLite read, so the harness stays offline either way.
    """
    entry = _load_json(os.path.join(CACHE_DIR, f"{symbol}_{kind}_annual.json"))
    if not entry:
        return None
    raw = entry.get("data_df_json")
    if not raw or raw == "{}":
        return None
    try:
        df = pd.read_json(StringIO(raw), orient="split")
    except Exception:
        return None
    if df.empty:
        return None
    merged = market_data._with_edgar_history(symbol, kind, "annual", df)
    return None if merged is None or merged.empty else merged


def _load_info(symbol: str) -> Optional[dict]:
    entry = _load_json(os.path.join(CACHE_DIR, f"{symbol}.json"))
    if not entry:
        return None
    info = entry.get("ticker_info") or entry.get("data")
    return info if isinstance(info, dict) and info else None


def discover_symbols(limit: Optional[int] = None) -> List[str]:
    """Symbols holding an info blob plus all three annual statements."""
    if not os.path.isdir(CACHE_DIR):
        raise SystemExit(f"No fundamentals cache at {CACHE_DIR}")
    stems = {
        f[: -len("_financials_annual.json")]
        for f in os.listdir(CACHE_DIR)
        if f.endswith("_financials_annual.json")
    }
    symbols = sorted(
        s
        for s in stems
        if os.path.exists(os.path.join(CACHE_DIR, f"{s}.json"))
        and os.path.exists(os.path.join(CACHE_DIR, f"{s}_balance_sheet_annual.json"))
        and os.path.exists(os.path.join(CACHE_DIR, f"{s}_cashflow_annual.json"))
    )
    return symbols[:limit] if limit else symbols


def evaluate(symbols: List[str], iterations: int = 2000) -> List[Dict[str, Any]]:
    """Run the live model over each symbol; one row per symbol."""
    rows: List[Dict[str, Any]] = []
    for i, sym in enumerate(symbols):
        info = _load_info(sym)
        if not info:
            continue
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price or price <= 0:
            continue

        try:
            res = fr.get_comprehensive_intrinsic_value(
                info,
                _load_statement(sym, "financials"),
                _load_statement(sym, "balance_sheet"),
                _load_statement(sym, "cashflow"),
                iterations=iterations,
            )
        except Exception as exc:  # a model that raises is itself a finding
            rows.append({"symbol": sym, "price": price, "crashed": str(exc)})
            continue

        models = res.get("models", {}) or {}
        dcf = models.get("dcf", {}) or {}
        graham = models.get("graham", {}) or {}
        epv = models.get("epv", {}) or {}
        avg = res.get("average_intrinsic_value")

        rows.append(
            {
                "symbol": sym,
                "price": float(price),
                "quote_type": (info.get("quoteType") or "").upper(),
                "sector": info.get("sector"),
                "avg_iv": avg,
                "mos_pct": res.get("margin_of_safety_pct"),
                "dcf_iv": dcf.get("intrinsic_value"),
                "dcf_model": dcf.get("model"),
                "dcf_error": dcf.get("error"),
                "graham_iv": graham.get("intrinsic_value"),
                "graham_model": graham.get("model"),
                "graham_error": graham.get("error"),
                "epv_iv": epv.get("intrinsic_value"),
                "epv_model": epv.get("model"),
                "epv_error": epv.get("error"),
                "status": res.get("valuation_status"),
                "spread_pct": res.get("model_spread_pct"),
                "note": res.get("valuation_note"),
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(symbols)}", file=sys.stderr)
    return rows


def _finite(values) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)] if arr.size else arr


def _dist(name: str, values, unit: str = "") -> Optional[Dict[str, Any]]:
    arr = _finite(values)
    if arr.size == 0:
        print(f"{name:<28} (no data)")
        return None
    pcts = {f"p{p}": float(np.percentile(arr, p)) for p in PCTS}
    body = "  ".join(f"p{p}={pcts[f'p{p}']:>9.1f}" for p in PCTS)
    print(f"{name:<28} n={arr.size:<6} {body} {unit}")
    return {"n": int(arr.size), **pcts}


def report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    crashed = [r for r in rows if r.get("crashed")]
    live = [r for r in rows if not r.get("crashed")]
    equities = [r for r in live if r.get("quote_type") == "EQUITY"]

    print("\n" + "=" * 96)
    print(f"INTRINSIC VALUE CALIBRATION — {total} symbols ({len(equities)} equities)")
    print("=" * 96)
    if crashed:
        print(f"\n!! {len(crashed)} symbols raised: {crashed[0]['crashed'][:80]}")

    def cov(rs, key, label):
        got = sum(1 for r in rs if r.get(key) is not None)
        print(f"  {label:<26} {got:>5}/{len(rs)}  ({got / max(1, len(rs)):>6.1%})")
        return got / max(1, len(rs))

    print("\nCOVERAGE (equities)")
    coverage = {
        "dcf": cov(equities, "dcf_iv", "DCF"),
        "epv": cov(equities, "epv_iv", "EPV"),
        "graham": cov(equities, "graham_iv", "Graham"),
        "average": cov(equities, "avg_iv", "Blended IV"),
    }

    statuses: Dict[str, int] = {}
    for r in equities:
        statuses[r.get("status") or "-"] = statuses.get(r.get("status") or "-", 0) + 1
    if len(statuses) > 1 or "-" not in statuses:
        print("\nVALUATION STATUS")
        for k, v in sorted(statuses.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<26} {v:>5}  ({v / max(1, len(equities)):>6.1%})")

    print("\nMARGIN OF SAFETY, % ((IV - price) / price). Well-calibrated => p50 near 0")
    mos = _dist("  blended (shipped)", [r.get("mos_pct") for r in equities])

    def as_mos(r, key):
        iv = r.get(key)
        return None if iv is None else (iv - r["price"]) / r["price"] * 100.0

    dcf_mos = _dist("  DCF alone", [as_mos(r, "dcf_iv") for r in equities])
    epv_mos = _dist("  EPV alone", [as_mos(r, "epv_iv") for r in equities])
    graham_mos = _dist("  Graham alone", [as_mos(r, "graham_iv") for r in equities])

    print("\nMODEL DISPERSION — |DCF - Graham| / price, %")
    both = [r for r in equities if r.get("dcf_iv") is not None and r.get("graham_iv") is not None]
    spread = _dist(
        "  spread",
        [abs(r["dcf_iv"] - r["graham_iv"]) / r["price"] * 100.0 for r in both],
    )

    print("\nMODEL VARIANTS USED")
    variants: Dict[str, int] = {}
    for r in equities:
        for key in ("dcf_model", "graham_model", "epv_model"):
            if r.get(key):
                variants[r[key]] = variants.get(r[key], 0) + 1
    for k, v in sorted(variants.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34} {v:>5}")

    print("\nTOP FAILURE REASONS")
    errors: Dict[str, int] = {}
    for r in equities:
        for key in ("dcf_error", "graham_error", "epv_error"):
            if r.get(key):
                errors[r[key][:60]] = errors.get(r[key][:60], 0) + 1
    for k, v in sorted(errors.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:<60} {v:>5}")

    # An IV more than 10x or under a tenth of price is not a valuation, it is a
    # data escape. Counted separately because percentiles hide the tails that
    # actually reach the UI. The tolerance keeps values sitting exactly on the
    # clamp boundary from being reported as escapes by float representation.
    tol = 1e-9
    absurd = [
        r
        for r in equities
        if r.get("avg_iv")
        and (
            r["avg_iv"] > 10 * r["price"] * (1 + tol)
            or r["avg_iv"] < 0.1 * r["price"] * (1 - tol)
        )
    ]
    print(f"\nABSURD (IV >10x or <0.1x price): {len(absurd)}/{len(equities)} "
          f"({len(absurd) / max(1, len(equities)):.1%})")
    for r in sorted(absurd, key=lambda r: -(r["avg_iv"] / r["price"]))[:5]:
        print(f"    {r['symbol']:<7} price={r['price']:>9.2f} iv={r['avg_iv']:>12.2f} "
              f"({r['avg_iv'] / r['price']:>7.1f}x)")

    return {
        "n_total": total,
        "n_equities": len(equities),
        "coverage": coverage,
        "mos_blended": mos,
        "mos_dcf": dcf_mos,
        "mos_epv": epv_mos,
        "mos_graham": graham_mos,
        "spread": spread,
        "absurd_frac": len(absurd) / max(1, len(equities)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap symbols (quick pass)")
    ap.add_argument("--iterations", type=int, default=2000, help="Monte Carlo draws")
    ap.add_argument("--json", type=str, default=None, help="write summary JSON here")
    ap.add_argument("--rows", type=str, default=None, help="write per-symbol CSV here")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    np.random.seed(7)  # MC draws must not move the headline between runs

    symbols = discover_symbols(args.limit)
    print(f"Evaluating {len(symbols)} cached symbols...", file=sys.stderr)
    rows = evaluate(symbols, iterations=args.iterations)
    summary = report(rows)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nSummary -> {args.json}")
    if args.rows:
        pd.DataFrame(rows).to_csv(args.rows, index=False)
        print(f"Rows    -> {args.rows}")


if __name__ == "__main__":
    main()
