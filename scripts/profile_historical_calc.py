#!/usr/bin/env python3
"""Profile the historical performance calculation pipeline.

Usage:
    python scripts/profile_historical_calc.py [username]

Defaults to 'kitmatan'. Produces:
  - scripts/profile_results/cprofile_stats.txt   (top-50 cumulative callers)
  - scripts/profile_results/cprofile_stats.prof  (full pstats dump; view with snakeviz)
  - scripts/profile_results/phase_timings.txt    (phase-level wall-clock breakdown)
"""
import cProfile
import io
import os
import pstats
import sys
import time
from datetime import date

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import config  # noqa: E402
from data_loader import load_and_clean_transactions  # noqa: E402
from portfolio_history import calculate_historical_performance  # noqa: E402

# ──────────────────────────────────────────────────────────────
# Monkey-patch the major pipeline phases so we can capture
# wall-clock timings *without* editing the main source code.
# ──────────────────────────────────────────────────────────────
_phase_timings: list[tuple[str, float]] = []

def _timed_wrapper(label, original_fn):
    """Return a wrapper that records wall-clock time for `original_fn`."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _phase_timings.append((label, elapsed))
        return result
    return wrapper

# Patch individual heavy functions
import portfolio_history as ph  # noqa: E402
import portfolio_valuation_kernels as pvk  # noqa: E402
import market_data as md  # noqa: E402

# Phase 1: Input preparation
ph._prepare_historical_inputs = _timed_wrapper(
    "Phase 1: _prepare_historical_inputs", ph._prepare_historical_inputs
)
# Phase 3: Market data fetch (historical prices)
_original_get_hist = md.MarketDataProvider.get_historical_data
md.MarketDataProvider.get_historical_data = _timed_wrapper(
    "Phase 3: MarketDataProvider.get_historical_data", _original_get_hist
)
# Phase 3b: FX data fetch
_original_get_fx = md.MarketDataProvider.get_historical_fx_rates
md.MarketDataProvider.get_historical_fx_rates = _timed_wrapper(
    "Phase 3b: MarketDataProvider.get_historical_fx_rates", _original_get_fx
)
# Phase 5: Daily results calculation
ph._load_or_calculate_daily_results = _timed_wrapper(
    "Phase 5: _load_or_calculate_daily_results", ph._load_or_calculate_daily_results
)
# Phase 7: Resampling & TWR
ph._calculate_accumulated_gains_and_resample = _timed_wrapper(
    "Phase 7: _calculate_accumulated_gains_and_resample",
    ph._calculate_accumulated_gains_and_resample,
)
# Numba kernels
pvk._calculate_daily_holdings_chronological_numba = _timed_wrapper(
    "Kernel: _calculate_daily_holdings_chronological_numba",
    pvk._calculate_daily_holdings_chronological_numba,
)
pvk._calculate_holdings_numba = _timed_wrapper(
    "Kernel: _calculate_holdings_numba",
    pvk._calculate_holdings_numba,
)
# Price unadjustment (often large loop)
ph._unadjust_prices = _timed_wrapper(
    "Phase 2: _unadjust_prices", ph._unadjust_prices
)

# ──────────────────────────────────────────────────────────────
# Load data and run
# ──────────────────────────────────────────────────────────────
def main():
    username = sys.argv[1] if len(sys.argv) > 1 else "kitmatan"
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, username)
    db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)

    if not os.path.exists(db_path):
        print(f"ERROR: No portfolio DB found at {db_path}")
        sys.exit(1)

    # Read gui_config for account_currency_map
    config_dir = os.path.join(user_data_dir, config.CONFIG_DIR)
    gui_config = {}
    gui_config_path = os.path.join(config_dir, config.GUI_CONFIG_FILENAME)
    if os.path.exists(gui_config_path):
        import json
        with open(gui_config_path) as f:
            gui_config = json.load(f)

    account_currency_map = {"SET": "THB"}
    account_currency_map.update(gui_config.get("account_currency_map", {}))
    account_cash_mode_map = dict(gui_config.get("account_cash_mode_map", {}))

    # Load transactions
    print(f"Loading transactions from {db_path}...")
    df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
        source_path=db_path,
        account_currency_map=account_currency_map,
        default_currency=config.DEFAULT_CURRENCY,
        is_db_source=True,
    )
    print(f"Loaded {len(df)} transactions.")

    if df.empty:
        print("ERROR: No transactions found.")
        sys.exit(1)

    # Parameters matching a typical dashboard load
    min_date = df["Date"].min().date()
    today = date.today()

    print(f"Date range: {min_date} → {today}")
    print("Display currency: USD")
    print()

    # ── cProfile run ──
    profiler = cProfile.Profile()
    _phase_timings.clear()

    print("=" * 72)
    print("PROFILING: calculate_historical_performance (caches DISABLED)")
    print("=" * 72)

    t_wall_start = time.perf_counter()
    profiler.enable()

    daily_df, prices, fx, status = calculate_historical_performance(
        all_transactions_df_cleaned=df,
        original_transactions_df_for_ignored=df,
        ignored_indices_from_load=ignored_indices,
        ignored_reasons_from_load=ignored_reasons,
        start_date=min_date,
        end_date=today,
        interval="1d",
        benchmark_symbols_yf=[],
        display_currency="USD",
        account_currency_map=account_currency_map,
        default_currency=config.DEFAULT_CURRENCY,
        use_raw_data_cache=False,      # Force fresh fetch
        use_daily_results_cache=False,  # Force full recalculation
        num_processes=1,                # Single-process for clean profiling
        include_accounts=None,
        worker_signals=None,
        user_symbol_map=dict(gui_config.get("user_symbol_map", {})),
        manual_overrides_dict=gui_config.get("manual_price_overrides", {}),
        user_excluded_symbols=set(gui_config.get("user_excluded_symbols", [])),
        original_csv_file_path=db_path,
        account_cash_mode_map=account_cash_mode_map,
    )

    profiler.disable()
    t_wall_end = time.perf_counter()

    # ── Save results ──
    out_dir = os.path.join(ROOT, "scripts", "profile_results")
    os.makedirs(out_dir, exist_ok=True)

    # Phase timings
    phase_file = os.path.join(out_dir, "phase_timings.txt")
    total_wall = t_wall_end - t_wall_start
    with open(phase_file, "w") as f:
        f.write(f"Total wall-clock time: {total_wall:.2f}s\n")
        f.write(f"Result rows: {len(daily_df)}\n")
        f.write(f"Status: {status[:120]}...\n")
        f.write(f"{'─' * 72}\n")
        f.write(f"{'Phase':<55} {'Time (s)':>8} {'%':>6}\n")
        f.write(f"{'─' * 72}\n")
        for label, elapsed in sorted(_phase_timings, key=lambda x: -x[1]):
            pct = 100 * elapsed / total_wall if total_wall else 0
            f.write(f"{label:<55} {elapsed:>8.2f} {pct:>5.1f}%\n")
        f.write(f"{'─' * 72}\n")

    # cProfile stats (text)
    stats_file = os.path.join(out_dir, "cprofile_stats.txt")
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.strip_dirs().sort_stats("cumulative").print_stats(50)
    with open(stats_file, "w") as f:
        f.write(stream.getvalue())

    # cProfile binary (for snakeviz / gprof2dot)
    prof_file = os.path.join(out_dir, "cprofile_stats.prof")
    profiler.dump_stats(prof_file)

    # ── Print summary ──
    print()
    print("=" * 72)
    print(f"TOTAL WALL-CLOCK TIME: {total_wall:.2f}s")
    print(f"Output rows: {len(daily_df)}")
    print("=" * 72)
    print()
    print("PHASE BREAKDOWN:")
    print(f"{'─' * 72}")
    print(f"{'Phase':<55} {'Time (s)':>8} {'%':>6}")
    print(f"{'─' * 72}")
    for label, elapsed in sorted(_phase_timings, key=lambda x: -x[1]):
        pct = 100 * elapsed / total_wall if total_wall else 0
        print(f"{label:<55} {elapsed:>8.2f} {pct:>5.1f}%")
    print(f"{'─' * 72}")
    print()
    print(f"Detailed results saved to: {out_dir}/")
    print(f"  • {phase_file}")
    print(f"  • {stats_file}")
    print(f"  • {prof_file}")
    print()
    print("To interactively explore the cProfile data:")
    print(f"  pip install snakeviz && snakeviz {prof_file}")


if __name__ == "__main__":
    main()
