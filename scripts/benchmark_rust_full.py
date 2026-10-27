#!/usr/bin/env python3
"""The whole Rust port against the Python it replaced, on real data.

Every ported area is timed on real inputs, Rust against the frozen original in
tests/reference/ (Numba warm for the three kernels, which were Numba before):

* Per user (default kitmatan and dheematan):
  - the replay kernels;
  - IRR and risk metrics;
  - FIFO lots;
  - the history valuation and accumulation;
  - the projection backtest;
  - the summary and history requests end to end, with *every* ported
    function swapped at once.
* Market-wide:
  - intrinsic values (Monte Carlo and deterministic models) for a set of real
    companies;
  - Buffett metrics over a sample of filers;
  - the ranking's value and quality scoring over the whole universe;
  - valuation bands and the track record;
  - the trend signal.
* Cold start: the first call in a fresh process.

Market data is fetched once and replayed (see benchmark_rust_core), so this
times computation, not the network. A/B timings are interleaved.

Usage: cd src && python ../scripts/benchmark_rust_full.py [username ...]
Needs numba (dev only) for the kernel and cold-start comparisons.
"""

import asyncio
import copy
import json
import logging
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import benchmark_rust_core as B  # noqa: E402  (sets up paths, taped fetches)

import pandas as pd  # noqa: E402

import portfolio_analyzer as PA  # noqa: E402
import portfolio_history as H  # noqa: E402
import portfolio_logic as PL  # noqa: E402
import projections as P  # noqa: E402
import risk_metrics as RM  # noqa: E402
from reference import fifo as ref_fifo  # noqa: E402
from reference import history_accumulation as ref_acc  # noqa: E402
from reference import history_valuation as ref_val  # noqa: E402
from reference import projections as ref_proj  # noqa: E402
from server.api import _calculate_historical_performance_internal  # noqa: E402

USERS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["kitmatan", "dheematan"]
IV_SYMBOLS = [
    "AAPL",
    "MSFT",
    "KO",
    "JPM",
    "O",
    "BRK-B",
    "XOM",
    "NVDA",
    "PG",
    "META",
    "COST",
    "V",
]
FILER_SAMPLE = 400

logging.disable(logging.WARNING)
fmt, timed = B.fmt, B.timed


def run_history_all(data):
    """The full-history view. benchmark_rust_core asks for period="max", which
    the service does not know and answers with its default one-year window."""
    return asyncio.run(
        _calculate_historical_performance_internal(
            currency="THB", period="all", accounts=None, benchmarks=[], data=data,
            return_df=True, interval="1d", force=True,
        )
    )  # fmt: skip


B.run_history = run_history_all  # capture() and the end-to-end runs use it


def ab(rust_fn, orig_fn, args=(), rounds=5, kwargs=None):
    """Interleaved medians: (rust, original)."""
    kwargs = kwargs or {}
    for fn in (rust_fn, orig_fn):  # warm both
        fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    samples = {"rust": [], "original": []}
    for r in range(rounds):
        order = [("rust", rust_fn), ("original", orig_fn)]
        for name, fn in order if r % 2 == 0 else reversed(order):
            a, k = copy.deepcopy(args), copy.deepcopy(kwargs)
            t0 = time.perf_counter()
            fn(*a, **k)
            samples[name].append(time.perf_counter() - t0)
    return statistics.median(samples["rust"]), statistics.median(samples["original"])


ROWS = []
SCOPE = {"user": None}


def row(area, what, rust_t, orig_t, base="Python"):
    ROWS.append(
        {
            "scope": SCOPE["user"] or "market",
            "area": area,
            "what": what,
            "rust": rust_t,
            "original": orig_t,
            "base": base,
        }
    )
    print(
        f"  {area:24s} {what:44s} rust {fmt(rust_t):>9s}  {base.lower()} {fmt(orig_t):>9s}  "
        f"x{orig_t / rust_t:,.1f}",
        flush=True,
    )


# Every function the port replaced on the summary/history path, both ways.
PORTED = {
    "fifo": ("calculate_fifo_lots_and_gains", [PA, PL], PA.calculate_fifo_lots_and_gains, ref_fifo.calculate_fifo_lots_and_gains),
    "val": ("_value_daily_holdings_vectorized", [H, PL], H._value_daily_holdings_vectorized, ref_val._value_daily_holdings_vectorized),
    "acc": ("_calculate_accumulated_gains_and_resample", [H, PL], H._calculate_accumulated_gains_and_resample, ref_acc._calculate_accumulated_gains_and_resample),
    "risk": ("calculate_all_risk_metrics", [RM], RM.calculate_all_risk_metrics, B.ref_risk.calculate_all_risk_metrics),
}  # fmt: skip


def install_all(rust, numba_impl):
    B.install(B.RUST if rust else numba_impl)
    for name, mods, new_fn, old_fn in PORTED.values():
        for m in mods:
            if hasattr(m, name):
                setattr(m, name, new_fn if rust else old_fn)


def record(data):
    """Real arguments of each Phase 6 function, from one summary and history."""
    seen = {}

    def spy(key, fn):
        def wrapped(*a, **k):
            seen.setdefault(key, (copy.deepcopy(a), copy.deepcopy(k)))
            return fn(*a, **k)

        return wrapped

    for key, (name, mods, new_fn, _) in PORTED.items():
        for m in mods:
            if hasattr(m, name):
                setattr(m, name, spy(key, new_fn))
    B.run_summary(data)
    run_history_all(data)
    for name, mods, new_fn, _ in PORTED.values():
        for m in mods:
            if hasattr(m, name):
                setattr(m, name, new_fn)
    return seen


def per_user(username, numba_impl):
    SCOPE["user"] = username
    data = B.user_data(username)
    print(f"\n== {username} ==", flush=True)
    args = B.capture(data)
    if "holdings" not in args and "daily" in args:
        args["holdings"] = B.holdings_args_from_daily(args["daily"])
    for name in B.KERNELS:
        if name in args:
            a = args[name]
            numba_impl[name](*a)
            row(
                "replay kernel",
                f"{name}: {B.shape_note(name, a)}",
                timed(B.RUST[name], a, 20),
                timed(numba_impl[name], a, 20),
                "Numba",
            )

    calls = args["irr_calls"]

    def all_irr(fn):
        for d, f in calls:
            fn(list(d), list(f))

    rs = statistics.median([timed(all_irr, (PA.calculate_irr,), 1) for _ in range(5)])
    py = statistics.median(
        [timed(all_irr, (B.ref_finance.calculate_irr,), 1) for _ in range(3)]
    )
    row("IRR", f"{len(calls):,} per-holding IRRs (one summary)", rs, py)

    hist = args["history"]
    series = hist.set_index("date")["value"]
    row(
        "risk metrics",
        f"{len(series):,}-day value series",
        *ab(
            RM.calculate_all_risk_metrics,
            B.ref_risk.calculate_all_risk_metrics,
            (series,),
            9,
        ),
    )

    seen = record(data)
    for key, label in (
        ("fifo", "FIFO lots and realized gains"),
        ("val", "daily holdings valuation"),
        ("acc", "TWR accumulation"),
    ):
        if key in seen:
            a, k = seen[key]
            _, _, new_fn, old_fn = PORTED[key]
            if key == "fifo":
                what = f"{label}: {len(k['transactions_df']):,} rows"
            elif key == "val":
                what = (
                    f"{label}: {'x'.join(map(str, k['daily_holdings_qty_np'].shape))}"
                )
            else:
                what = f"{label}: {len(k['daily_df']):,} days"
            row("portfolio engine", what, *ab(new_fn, old_fn, a, 5, k))

    proj = asyncio.run(
        _calculate_historical_performance_internal(
            currency="USD", period="all", accounts=None, benchmarks=["SPY"], data=data,
            return_df=True, interval="1d", force=True,
        )
    )  # fmt: skip
    df = proj[0] if isinstance(proj, tuple) else proj
    if isinstance(df, pd.DataFrame) and "Portfolio Accumulated Gain" in df.columns:
        if "date" in df.columns:
            df = df.set_index("date")
        twr = df["Portfolio Accumulated Gain"].dropna()
        bench = (
            df["SPY Accumulated Gain"] if "SPY Accumulated Gain" in df.columns else None
        )
        value = (
            df["Portfolio Value"]
            if "Portfolio Value" in df.columns
            else df.get("value")
        )
        row(
            "projections",
            f"backtest on {len(twr):,} days of TWR",
            *ab(
                P.backtest_projection,
                ref_proj.backtest_projection,
                (twr, bench, value),
                3,
            ),
        )
        row(
            "projections",
            "projection cone",
            *ab(
                P.compute_projection,
                ref_proj.compute_projection,
                (twr, float(value.dropna().iloc[-1])),
                9,
            ),
        )

    for label, fn in (("summary", B.run_summary), ("history", run_history_all)):
        samples = {"rust": [], "original": []}
        for impl in (True, False):
            install_all(impl, numba_impl)
            fn(data)
        for r in range(5):
            for rust in (True, False) if r % 2 == 0 else (False, True):
                install_all(rust, numba_impl)
                samples["rust" if rust else "original"].append(timed(fn, (data,), 1))
        install_all(True, numba_impl)
        row(
            "end to end",
            f"/{label} computation, everything swapped",
            statistics.median(samples["rust"]),
            statistics.median(samples["original"]),
            "Original",
        )

    cold = B.cold_start({k: args[k] for k in B.KERNELS if k in args})
    row(
        "cold start",
        "first kernel calls in a fresh process",
        cold["rust"]["total"],
        cold["numba"]["total"],
        "Numba",
    )


def market_wide():
    SCOPE["user"] = None
    print("\n== market-wide ==", flush=True)
    import buffett_metrics as BM
    import buffett_pipeline as BP
    import buffett_rank as BR
    import buffett_store as BS
    import buffett_value as BV
    import edgar_provider
    import edgar_sic
    import financial_ratios as FR
    import strategies as ST
    import track_record as TR
    import valuation_history as VH
    from reference import buffett_metrics as ref_bm
    from reference import buffett_rank as ref_br
    from reference import buffett_value as ref_bv
    from reference import financial_ratios as ref_fr
    from reference import strategies as ref_st
    from reference import track_record as ref_tr
    from reference import valuation_history as ref_vh
    from portfolio_logic import get_shared_mdp

    mdp = get_shared_mdp()
    t = {"rust": 0.0, "original": 0.0}
    for sym in IV_SYMBOLS:
        rs, py = ab(
            lambda s=sym: FR.get_intrinsic_value_for_symbol(s, mdp, force_refresh=True),
            lambda s=sym: ref_fr.get_intrinsic_value_for_symbol(
                s, mdp, force_refresh=True
            ),
            rounds=3,
        )
        t["rust"] += rs
        t["original"] += py
    row(
        "intrinsic value",
        f"{len(IV_SYMBOLS)} companies, all twelve models",
        t["rust"],
        t["original"],
    )

    filers = BP._resolve_universe()
    sample = filers[:: max(1, len(filers) // FILER_SAMPLE)]
    sic_map = edgar_sic.get_sic_map()
    jobs = []
    for cik, entry in sample:
        sic = sic_map.get(cik)
        model = (
            edgar_sic.model_for_sic(sic)
            if sic is not None
            else BP.infer_model_from_facts(cik)
        )
        concepts = edgar_provider.get_concept_values(cik)
        if not concepts or "net_income" not in concepts:
            continue
        ratios = BM._ratios_by_period(cik)
        if model in ("bank", "insurer", "reit"):
            from edgar_concepts import BANK_CONCEPTS, REIT_CONCEPTS

            sector = edgar_provider.get_concept_values(
                cik, list(BANK_CONCEPTS if model != "reit" else REIT_CONCEPTS)
            )
            name = (
                "_compute_reit_metrics" if model == "reit" else "_compute_bank_metrics"
            )
            jobs.append((name, (concepts, sector, ratios)))
        else:
            jobs.append(("_compute_generic_metrics", (concepts, ratios)))

    def run_jobs(mod):
        for name, a in jobs:
            getattr(mod, name)(*a)

    row(
        "Buffett metrics",
        f"{len(jobs)} real filers (arithmetic only)",
        *ab(lambda: run_jobs(BM), lambda: run_jobs(ref_bm), rounds=3),
    )

    frame = BS.get_store().get_scores_frame().set_index("symbol", drop=False)
    row(
        "ranking",
        f"value scoring, {len(frame):,} ranked companies",
        *ab(BV.score_value, ref_bv.score_value, (frame,), 5),
    )
    t0 = time.perf_counter()
    companies = BP.collect_metrics(filers)
    print(
        f"  (metrics for all {len(companies):,} filers collected in {time.perf_counter() - t0:,.0f} s)",
        flush=True,
    )
    row(
        "ranking",
        f"quality scoring, {len(companies):,} filers",
        *ab(BR.score_quality, ref_br.score_quality, (companies,), 5),
    )

    ranked = [(c.symbol, c.cik) for c in companies if c.passes_gates][:60]
    row(
        "valuation bands",
        f"{len(ranked)} companies",
        *ab(
            lambda: [VH.bands(s, c) for s, c in ranked],
            lambda: [ref_vh.bands(s, c) for s, c in ranked],
            rounds=3,
        ),
    )
    row(
        "track record",
        f"{len(ranked)} companies",
        *ab(
            lambda: [TR.build(s, c) for s, c in ranked],
            lambda: [ref_tr.build(s, c) for s, c in ranked],
            rounds=3,
        ),
    )

    from utils_time import get_est_today

    today = get_est_today()  # the market's day, not the server's
    prices = ST.load_signal_prices("SPY", today)
    if isinstance(prices, pd.Series) and len(prices):
        row(
            "trend signal",
            f"SPY, {len(prices):,} closes, 10-month SMA",
            *ab(
                ST.evaluate_trend_signal,
                ref_st.evaluate_trend_signal,
                (prices, 10, today),
                9,
            ),
        )


def main():
    numba_impl = B.jit_reference()
    for u in USERS:
        per_user(u, numba_impl)
    if "--no-market" not in sys.argv:
        market_wide()
    out = os.path.join(ROOT, "profiles", "benchmark_rust_full.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(ROWS, f, indent=2, default=str)
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    main()
