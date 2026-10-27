#!/usr/bin/env python3
"""Benchmark the Rust calculation core against the Python/Numba code it replaced.

Three measurements, all on a real user's ledger:

1. Kernels: each kernel on the exact arrays a real summary/history request
   hands it (captured from a live run), Rust vs Numba (JIT warm) vs plain
   Python. The Numba and plain-Python versions are the frozen originals in
   tests/reference/numba_kernels.py, compiled with their production decorators.
2. Cold start: the first call in a fresh process, which for Numba includes JIT
   compilation with an empty cache (what every edit or fresh deploy pays).
3. End to end: the cache-bypassing /history and /summary computations, with
   every ported function swapped between Rust and its original (Numba kernels,
   Python IRR), same process. Market
   data is fetched once and replayed, so these time computation, not network.

Usage: cd src && python ../scripts/benchmark_rust_core.py [username ...]
Default users: kitmatan dheematan. Needs numba installed (dev only).
"""

import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(SRC)

import numba  # noqa: E402

import config  # noqa: E402

config.get_app_cache_dir = lambda: None  # no L1 holdings cache: measure the work

import market_data  # noqa: E402
import portfolio_analyzer as PA  # noqa: E402
import portfolio_history as H  # noqa: E402
import portfolio_logic as PL  # noqa: E402
import portfolio_valuation_kernels as K  # noqa: E402
from portfolio_logic import calculate_portfolio_summary  # noqa: E402
from reference import numba_kernels as ref  # noqa: E402
from reference import finance as ref_finance  # noqa: E402
from reference import risk_metrics as ref_risk  # noqa: E402
import risk_metrics as RM  # noqa: E402
from server.api import _calculate_historical_performance_internal  # noqa: E402
from server.auth import User  # noqa: E402
from server.dependencies import get_transaction_data  # noqa: E402

USERS = sys.argv[1:] or ["kitmatan", "dheematan"]

# Market data is recorded on first fetch and replayed after, so the end-to-end
# timings measure computation rather than the network, and every run sees the
# same prices.
_real_fetch = market_data._run_isolated_fetch
_tape = {}


def _taped_fetch(*args, **kwargs):
    import copy
    import pickle

    key = pickle.dumps((args, sorted(kwargs.items())))
    if key not in _tape:
        _tape[key] = _real_fetch(*args, **kwargs)
    return copy.deepcopy(_tape[key])


market_data._run_isolated_fetch = _taped_fetch
KERNELS = ("holdings", "daily", "core")

RUST = {
    "holdings": K._calculate_holdings_numba,
    "daily": H._calculate_daily_holdings_chronological_numba,
    "core": PA._process_numba_core,
    "irr": PA.calculate_irr,
}
PLAIN = {
    "holdings": ref._calculate_holdings_numba,
    "daily": ref._calculate_daily_holdings_chronological_numba,
    "core": ref._process_numba_core,
}


def jit_reference():
    """The originals under their production decorators, compiled once.

    Numba resolves globals from the defining module, so the jitted copy gets a
    module of its own; `ref` stays plain Python for the plain-Python timings.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "numba_reference", os.path.join(ROOT, "tests", "reference", "numba_kernels.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fast = numba.jit(nopython=True, fastmath=True, cache=False)
    mod._dividend_amount = fast(mod._dividend_amount)
    return {
        "holdings": fast(mod._calculate_holdings_numba),
        "daily": fast(mod._calculate_daily_holdings_chronological_numba),
        "core": numba.jit(nopython=True, cache=False)(mod._process_numba_core),
        "irr": ref_finance.calculate_irr,
    }


def install(impl):
    K._calculate_holdings_numba = impl["holdings"]
    H._calculate_daily_holdings_chronological_numba = impl["daily"]
    PL._calculate_daily_holdings_chronological_numba = impl["daily"]
    PA._process_numba_core = impl["core"]
    PA.calculate_irr = impl["irr"]


def user_data(username):
    return get_transaction_data(
        User(id=1, username=username, alias=None, is_active=True, created_at="")
    )


def run_summary(data):
    df, overrides, symmap, excluded, acc_ccy, cash_modes, _, _ = data
    return calculate_portfolio_summary(
        all_transactions_df_cleaned=df,
        original_transactions_df_for_ignored=df,
        ignored_indices_from_load=set(),
        ignored_reasons_from_load={},
        fmp_api_key=None,
        display_currency="THB",
        show_closed_positions=True,
        manual_overrides_dict=overrides,
        user_symbol_map=symmap,
        user_excluded_symbols=excluded,
        include_accounts=None,
        account_currency_map=acc_ccy,
        default_currency=config.DEFAULT_CURRENCY,
        account_interest_rates={},
        interest_free_thresholds={},
        account_cash_mode_map=cash_modes,
        db_mtime=0,
    )


def run_history(data):
    return asyncio.run(
        _calculate_historical_performance_internal(
            currency="THB",
            period="all",
            accounts=None,
            benchmarks=[],
            data=data,
            return_df=True,
            interval="1d",
            force=True,
        )
    )


def capture(data):
    """The arguments of the largest call each kernel receives in a real run."""
    seen = {}

    def recorder(name, fn, size_of):
        def wrapped(*args):
            if name not in seen or size_of(args) > size_of(seen[name]):
                seen[name] = args
            return fn(*args)

        return wrapped

    irr_calls = seen.setdefault("irr_calls", [])
    rust_irr = PA.calculate_irr

    def irr_recorder(dates, flows):
        irr_calls.append((list(dates), list(flows)))
        return rust_irr(dates, flows)

    install(
        {
            "holdings": recorder("holdings", RUST["holdings"], lambda a: len(a[1])),
            "daily": recorder("daily", RUST["daily"], lambda a: len(a[0]) * len(a[1])),
            "core": recorder("core", RUST["core"], lambda a: len(a[0])),
            "irr": irr_recorder,
        }
    )
    run_summary(data)
    seen["history"] = run_history(data)
    install(RUST)
    return seen


def holdings_args_from_daily(d):
    """Single-date kernel arguments for the ledger a chronological call saw."""
    (days, dates, syms, to_accs, accs, types, qty, comms, splits, price, totals,
     ns, na, split, ssplit, buy, dep, sell, wd, ss, btc, tr, so, div, intr, fees,
     tax, cash, tol, shortable, modes) = d  # fmt: skip
    import numpy as np

    ccys = np.zeros(len(dates), dtype=np.int64)
    return (int(days[-1]), dates, syms, accs, to_accs, types, qty, price, totals,
            comms, splits, ccys, ns, na, 1, split, ssplit, buy, dep, sell, wd, ss,
            btc, tr, so, fees, div, intr, tax, cash, tol, shortable, modes)  # fmt: skip


def timed(fn, args, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def shape_note(name, args):
    if name == "holdings":
        return f"{len(args[1]):,} tx, {args[12]} symbols x {args[13]} accounts"
    if name == "daily":
        return f"{len(args[0]):,} days x {len(args[1]):,} tx, {args[11]} x {args[12]}"
    return f"{len(args[0]):,} tx, {args[15]} symbols x {args[16]} accounts"


COLD_START = r"""
import sys, os, time, json
sys.path.insert(0, {src!r}); sys.path.insert(0, {tests!r}); os.chdir({src!r})
import numpy as np, pickle
args = pickle.load(open({pkl!r}, "rb"))
t0 = time.perf_counter()
if {which!r} == "rust":
    import investa_core
    fns = dict(holdings=investa_core.calculate_holdings,
               daily=investa_core.calculate_daily_holdings_chronological,
               core=investa_core.process_core)
else:
    import numba
    from reference import numba_kernels as ref
    fast = numba.jit(nopython=True, fastmath=True, cache=True)
    ref._dividend_amount = fast(ref._dividend_amount)
    fns = dict(holdings=fast(ref._calculate_holdings_numba),
               daily=fast(ref._calculate_daily_holdings_chronological_numba),
               core=numba.jit(nopython=True, cache=True)(ref._process_numba_core))
out = {{}}
for name in args:
    t = time.perf_counter(); fns[name](*args[name]); out[name] = time.perf_counter() - t
out["total"] = time.perf_counter() - t0
print(json.dumps(out))
"""


def cold_start(args_by_kernel):
    """First call in a fresh process; Numba with an empty JIT cache."""
    import pickle

    with tempfile.TemporaryDirectory() as tmp:
        pkl = os.path.join(tmp, "args.pkl")
        # The Rust shims cast to the exact dtypes; feed both sides the same.
        with open(pkl, "wb") as f:
            pickle.dump(args_by_kernel, f)
        result = {}
        for which in ("rust", "numba"):
            env = dict(os.environ, NUMBA_CACHE_DIR=os.path.join(tmp, "nbcache"))
            code = COLD_START.format(
                src=SRC, tests=os.path.join(ROOT, "tests"), pkl=pkl, which=which
            )
            out = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env
            )
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-2000:])
            result[which] = json.loads(out.stdout.strip().splitlines()[-1])
        return result


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:,.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:,.1f} ms"
    return f"{seconds:,.2f} s"


def main():
    numba_impl = jit_reference()
    report = {"users": {}}
    for username in USERS:
        data = user_data(username)
        args = capture(data)
        if "holdings" not in args and "daily" in args:
            # The single-date kernel is not on the summary/history path (they
            # use the chronological one), so run it on the same real ledger,
            # valued as of its last day.
            args["holdings"] = holdings_args_from_daily(args["daily"])
        missing = [k for k in KERNELS if k not in args]
        if missing:
            print(f"{username}: kernels never called: {missing}")
        u = report["users"][username] = {"kernels": {}, "end_to_end": {}}

        print(f"\n== {username}: kernels (median, JIT warm) ==")
        for name in KERNELS:
            if name not in args:
                continue
            a = args[name]
            numba_impl[name](*a)  # compile outside the timing
            reps = 20
            rust_t = timed(RUST[name], a, reps)
            numba_t = timed(numba_impl[name], a, reps)
            plain_t = timed(PLAIN[name], a, 1 if name == "daily" else 3)
            u["kernels"][name] = {
                "shape": shape_note(name, a),
                "rust": rust_t,
                "numba": numba_t,
                "python": plain_t,
            }
            print(
                f"  {name:9s} {shape_note(name, a):42s} rust {fmt(rust_t):>9s}  "
                f"numba {fmt(numba_t):>9s}  python {fmt(plain_t):>9s}  "
                f"rust/numba x{numba_t / rust_t:,.2f}  python/rust x{plain_t / rust_t:,.0f}"
            )

        print(f"== {username}: Phase 2 — IRR and risk metrics (median) ==")
        calls = args["irr_calls"]

        def all_irr(fn):
            for d, f in calls:
                fn(list(d), list(f))

        import logging

        logging.disable(logging.WARNING)
        irr_py = statistics.median([timed(all_irr, (ref_finance.calculate_irr,), 1) for _ in range(3)])
        irr_rs = statistics.median([timed(all_irr, (PA.calculate_irr,), 1) for _ in range(5)])
        logging.disable(logging.NOTSET)
        u["irr"] = {"calls": len(calls), "python": irr_py, "rust": irr_rs}
        print(
            f"  IRR       {len(calls):,} calls in one summary          rust {fmt(irr_rs):>9s}  "
            f"python {fmt(irr_py):>9s}  python/rust x{irr_py / irr_rs:,.0f}"
        )
        hist = args["history"]
        series = hist.set_index("date")["value"] if "value" in hist.columns else None
        if series is not None and len(series) > 5:
            risk_py = timed(ref_risk.calculate_all_risk_metrics, (series,), 20)
            risk_rs = timed(RM.calculate_all_risk_metrics, (series,), 20)
            u["risk"] = {"days": len(series), "python": risk_py, "rust": risk_rs}
            print(
                f"  risk      {len(series):,}-day value series              rust {fmt(risk_rs):>9s}  "
                f"python {fmt(risk_py):>9s}  python/rust x{risk_py / risk_rs:,.1f}"
            )

        rounds = 5
        print(
            f"== {username}: end to end ({rounds} interleaved rounds, caches bypassed) =="
        )
        impls = {"rust": RUST, "original": numba_impl}
        for label, fn in (("summary", run_summary), ("history", run_history)):
            for impl in impls.values():  # warm both: data replayed, imports done
                install(impl)
                fn(data)
            samples = {name: [] for name in impls}
            for r in range(rounds):
                # Alternate the order so neither side always runs first.
                order = list(impls) if r % 2 == 0 else list(reversed(impls))
                for name in order:
                    install(impls[name])
                    samples[name].append(timed(fn, (data,), 1))
            install(RUST)
            res = {
                name: {"median": statistics.median(v), "min": min(v), "samples": v}
                for name, v in samples.items()
            }
            u["end_to_end"][label] = res
            rm, nm = res["rust"]["median"], res["original"]["median"]
            print(
                f"  {label:9s} rust {fmt(rm):>9s} (min {fmt(res['rust']['min'])})  "
                f"original {fmt(nm):>9s} (min {fmt(res['original']['min'])})  "
                f"median difference {fmt(abs(nm - rm))} "
                f"({'rust' if rm < nm else 'original'} faster)"
            )

        cold = cold_start({k: args[k] for k in KERNELS if k in args})
        u["cold_start"] = cold
        print(f"== {username}: cold start (fresh process, first call of each kernel) ==")
        print(
            f"  rust {fmt(cold['rust']['total'])}   numba {fmt(cold['numba']['total'])} "
            f"(of which JIT compile is nearly all)"
        )

    out = os.path.join(ROOT, "profiles", "benchmark_rust_core.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    main()
