# Rust Calculation Core: Port Plan and Status

**Status:** Phases 1–5 done (the Numba kernels; IRR/NPV and risk metrics; the intrinsic-value models; the Buffett ranking; projections and the reader-facing records). Phase 6 to do.
**Branch:** `feat/rust-core`. One long-lived branch, merged to `main` when the whole port is verified.
**Started:** 24 Oct 2026
**Scope:** every calculation module in `src/` moves into one Rust crate, `rust/investa_core`. Python calls it through PyO3. These stay Python:

- FastAPI routes, auth and caches.
- Data fetching: yfinance, Tiingo, FMP, SEC EDGAR, BOT and ECB.
- PDF statement parsing (pdfplumber), exchange calendars (pandas-market-calendars) and Gemini.

The API, the OpenAPI types and all three clients are untouched.

This reverses the recommendation in [go_rust_evaluation.md](go_rust_evaluation.md), by decision on 24 Oct 2026. That document's measurements still stand: whole requests are dominated by market-data sync, not calculation. So expect speed gains in the batch and cold-start paths, not in a typical request.

---

## How each module is ported

1. **Port line for line.** Each Python branch keeps its order of operations, so floating-point results are bit-identical rather than merely close.
   - Python's `min()` is reproduced exactly. It differs from `f64::min` when NaN is involved.
   - Negative indices wrap as they do in NumPy.
   - Quirks are preserved, not fixed. A fix is a separate, deliberate change.
2. **Freeze the original as an oracle.** The Python original moves to `tests/reference/` unchanged, apart from removing JIT decorators.
3. **Differential-test it.** Seeded random inputs reach every branch, and `tests/test_rust_*_parity.py` requires *bitwise* equality with the oracle.
4. **Check real data.** Both users' summary and history are computed with old and new code in one process, with market data recorded and replayed so both see identical prices.
5. **Swap in a thin shim.** The runtime module calls Rust under the original function name and argument order, so no call site changes.

## Building

The crate builds in four places:

- **CI:** `pip install ./rust/investa_core` runs after the Python requirements. A `rust-core` job runs `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`.
- **Docker:** the builder stage installs Rust and builds the wheel. The runtime image gets only the compiled module.
- **`./start_investa.sh`:** it runs `scripts/ensure_rust_core.py`, which rebuilds the module when it was built from older sources. It compares the hash `build.rs` bakes into the module with the hash of the sources on disk.
- **By hand:** `python3 scripts/ensure_rust_core.py`, or `maturin develop --release` in `rust/investa_core`.

`portfolio_version.CURRENT_HIST_VERSION` includes the crate's source hash, so changing a kernel invalidates cached histories, just as changing a Python engine file does.

## Phases

| Phase | Modules | Status |
|---|---|---|
| 1 | Replay kernels: `portfolio_valuation_kernels` (single-date and chronological holdings), `portfolio_analyzer._process_numba_core`. Numba removed. | **Done** |
| 2 | Financial math: `finutils` NPV/IRR (with SciPy's secant and Brent solvers), `risk_metrics` | **Done** |
| 3 | Intrinsic-value models in `financial_ratios`: (a) the twelve Monte Carlo models and their statistics; (b) WACC, the growth blend, FCF normalization, the projection models, EPV and the model blend | **Done** |
| 4 | Buffett ranking: `buffett_metrics`; the scoring in `buffett_rank` and `buffett_value` | **Done** |
| 5 | `projections`; the aggregations in `valuation_history`, `strategies` and `track_record` | **Done** |
| 6 | Pandas-heavy portfolio engine: (a) FIFO lots and realized gains; (b) daily holdings valuation; (c) TWR accumulation | **Done** |
| 7 | What the final benchmark left on the request path: (a) the historical FX lookup and the IRR/MWR cash flows; (b) statement assembly and the ratio history | **Done** |
| 8 | What a profile after Phase 7 left: the daily price read (Rust), plus three Python fixes: the NYSE calendar cache, per-company EDGAR read sessions, a cheaper UTC index | **Done** |
| 9 | The summary's per-holding and per-account filters, selected once (Python) | **Done** |
| 10 | Storage foundation: SQL in Rust on Python's own connections (`sql.rs`, `rust_sql.py`) | **Done** |
| 11 | Storage modules: (a) `db_utils`; (b) `market_db`; (c) `buffett_store`; (d) the EDGAR fact store and company session | **Done** |
| 12 | Import: `data_loader`, `csv_utils`, `io_handlers`, `db_utils.migrate_csv_to_db` | **Done** |
| 13 | Summary engine: (a) `corporate_actions` and the analyzer's standalone helpers; (b) the report extractors; (c) the holdings pass and FIFO; (d) the summary rows and totals; (e) `portfolio_logic` | **Done** |
| 14 | History engine: (a) `portfolio_cashflows` **done**; (b) the small history helpers; (c) input prep and daily valuation; (d) accumulation and the daily results; (e) `calculate_historical_performance` | In progress |
| 15 | Valuation and ranking: EDGAR statement assembly, the rest of `financial_ratios`, the `buffett_*`, `strategies`, `track_record` and `valuation_history` glue | Planned |
| 16 | Utilities: what is left of `finutils`, `risk_metrics` and the helpers | Planned |

## Phase 1 results (24 Oct 2026)

**Parity**

- Fuzz, 9,000 seeded ledgers: bit-identical to the original source run as Python.
- Against production Numba, the analyzer core is bit-identical. The two holdings kernels differ by at most 2.8 × 10⁻¹⁴ relative, because Numba's `fastmath` reorders floating-point arithmetic.
- Real data (kitmatan and dheematan, USD and THB): summary metrics, every holdings cell and the full history series are **exactly** equal, old vs new.
- Full suite: 1,145 tests pass.

**Speed** (`scripts/benchmark_rust_core.py`; raw numbers in `profiles/benchmark_rust_core.json`):

| | Rust | Numba (warm) | Plain Python |
|---|---|---|---|
| Daily holdings replay (8,885 days × ~5,000 tx) | 19.6–21.6 ms | 26.5–27.7 ms | 1.8 s |
| Analyzer replay | 64–69 µs | 106–140 µs | 15–17 ms |
| Single-date holdings | 37–39 µs | 33–37 µs | 7.2 ms |
| First call after an edit or fresh deploy | 46–64 ms | 5.0 s (JIT compile) | n/a |
| Full summary request | 2.05 s | 2.05–2.08 s | n/a |
| History request, default one-year window* | 4.2 s / 6.3 s | 4.1 s / 6.4 s | n/a |

\* The script asked for `period="max"`, which the service does not know, so it answered with its default one-year window. It now asks for `"all"`; the full-history numbers are in the final benchmark below.

End-to-end differences are 10–75 ms in both directions, which is noise. The real gain is cold start: the 5-second JIT compile on each edit or deploy is gone.

## Phase 2 results (24 Oct 2026)

**What moved:**
- `finutils.calculate_npv` and `calculate_irr`, ported line for line together with what they relied on:
  - SciPy 1.16's secant `newton` and C `brentq`, including the wrapper rule that a NaN function value aborts the search.
  - NumPy's `isclose`.
  - CPython 3.12's compensated `sum()`. The IRR's total-loss check uses it, and the result depends on whether each flow is an exact `float` or a NumPy scalar. Real ledgers pass both.
- All of `risk_metrics`, on exact ports of NumPy's pairwise sum and pandas' NaN-skipping mean, variance, min and cummax.
- What stayed Python: index alignment, `dropna`, `infer_periods_per_year` (which works on a pandas `DatetimeIndex`), and beta's covariance.

**Parity**
- Fuzz: IRR, NPV and every risk metric are bitwise equal to the frozen originals.
- Root finders compared directly with SciPy: 20,000 random brackets and iteration caps, every converge, raise and NaN case matching bit for bit.
- Real data (kitmatan, dheematan, atchamatan in USD and THB): every summary figure and every holdings cell is bitwise equal, including both IRR columns. All risk metrics and the benchmark scoreboard are bitwise equal too.
- **Rule adopted: BLAS/LAPACK-backed NumPy calls stay NumPy calls.** `np.cov`'s dot product (Accelerate here) sums in an order of its own; it matched neither sequential nor pairwise summation in 79% of trials. So `calculate_beta` keeps pandas' covariance and takes only its variance from Rust, and Beta and Alpha stay bitwise equal. The same rule covers `np.polyfit` (LAPACK) and `np.convolve` (BLAS `ddot`) in Phase 3.
- Full suite: 1,172 tests pass.

**SciPy fuses three operations on arm64.** The first `brentq` comparison was off by 1 ulp. The disassembly of SciPy's `_zeros.cpython-312-darwin.so` shows clang fused exactly three multiply-adds (`fmadd`/`fnmsub`):
- `delta`
- the extrapolation numerator
- the step bound `3|sbis| - delta`

The port uses `mul_add` at those three sites on aarch64 only. x86-64 wheels are built for baseline x86-64, which has no FMA, so nothing is fused there. On any other target, recheck it against the disassembly.

**Speed:** the IRR over the 772 per-holding calls one round of summaries makes (three users, two currencies) takes 886 ms in Python and 13 ms in Rust, 68× faster. That is about 148 ms → 2 ms per summary request, roughly 7% of one.

**Build fix: Xcode 27 linker.** `ld-27037` puts LINKEDIT's string pool directly after the indirect-symbol table, without padding. An odd number of 4-byte entries leaves the pool 4 bytes off an 8-byte boundary, and the macOS 27 loader rejects the binary ("mis-aligned LINKEDIT string pool"). The layout shifts as code changes, so earlier builds loaded by luck.
- No alternative linker works: `-ld_classic` is gone, and rust-lld cannot read the macOS 27 SDK's `.tbd` stubs.
- `rust/investa_core/.cargo/config.toml` routes macOS links through `build-support/macos_linker.py`. It links with `cc`, then, only when the pool is misaligned, drops the signature, inserts the missing padding, shifts the affected offsets, and ad-hoc signs again.
- The release profile sets `strip = false`, because rustc's post-link `strip -S` re-lays-out LINKEDIT and undoes the fix.

## Phase 3a results (25 Oct 2026): Monte Carlo models

**What moved:** `_generate_mc_stats` and the twelve `run_monte_carlo_*` models (`mc.rs`).
- **The draws stay NumPy's.** Each shim calls `np.random.normal` exactly as before, same order and arguments, and every operation that can raise stays on the same side of the draws. The models share one seeded stream per valuation, so drawing one sample more or less would shift every model after it.
- **Everything after the draw is Rust:** clipping, the faded projection with mid-year discounting, terminal values, and the percentiles, standard deviation and histogram.
- **`np.convolve` (the histogram smoothing) stays NumPy,** under the BLAS rule: its edges go through `ddot`.

**NumPy semantics the port had to reproduce**, each measured on the installed NumPy 2.0:
- **`array ** scalar` fast paths, by value:** an exponent of 2 squares, -1 takes the reciprocal, 0.5 the square root. Apple's `pow` is not correctly rounded for those, so the results differ from `pow`. Python's `float ** float` never takes a fast path. The port has `np_pow_scalar` for the first and `libm_pow` for the second, the latter kept out of LLVM's reach because LLVM folds a constant `pow(x, 2.0)` into `x * x`.
- **`np.clip` is `MIN(MAX(x, lo), hi)`,** NaN passing through, and `hi` wins when `lo > hi` (negative EPS).
- **`np.add.reduce` reduces in 8,192-element buffers:** `0.0 + pairwise(buffer 1) + pairwise(buffer 2) + …`. This was found here: a 10,000-sample standard deviation was 1 ulp off. It also corrects the Phase 2 reductions for series longer than 8,192 points, and negative zeros now sum to +0.0 as in NumPy.
- **`np.percentile`** uses NumPy's two-sided `_lerp`. **`np.histogram`** uses `linspace` edges and NumPy's index corrections.

**Parity**
- Seeded fuzz over all twelve models: projection lengths -1 to 10, negative EPS, missing or zero shares, NaN inputs, and 10 to 10,000 iterations. Results are bitwise equal, *and NumPy's generator ends in the same state*.
- Real data: a full intrinsic-value run for AAPL, KO, JPM, NVDA, MSFT, XOM, O and BRK-B has zero differences in the complete result, every model's Monte Carlo statistics, histogram and the blend included.
- Full suite: 1,180 tests pass.

## Phase 3b (25 Oct 2026): deterministic valuation arithmetic

**What moved** (`valuation.rs`):
- the scalar faded projection that DCF, DDM, D-CFO and D-NI share;
- `calculate_wacc`'s arithmetic;
- `blended_growth_estimate`'s median, mean, shrink and clamp;
- the plausible-margin filter and median of `estimate_fcf_margin` and `through_cycle_fcf_margin`;
- the medians in `normalized_base_fcf`, the EPV and `historical_mean_multiples` (with its percentiles);
- `_model_reliability`;
- `blend_intrinsic_values`' weighted mean, spread and confidence.

**The boundary.** Loops and aggregations moved. What stays Python:
- reading the statements and the company profile;
- choosing between methods, the result dictionaries and their notes;
- one-expression formulas such as Graham's `eps * (8.5 + 2g) * 4.4 / y` or Lynch's `eps * multiplier`. Moving a single expression behind a call adds an FFI hop and moves no logic;
- `calculate_key_ratios_timeseries`, which is per-period single divisions assembled into a DataFrame, with no aggregation to port;
- the log-linear growth regression, under the LAPACK rule (`np.polyfit`).

**Python scalar semantics reproduced:**
- `float ** x` and `np.float64 ** x` are always the C library's `pow`. NumPy *scalars* never take the array fast paths: 100,000 of 100,000 matched `pow`.
- `sum(list)` is CPython's compensated sum. A projection's terms are NumPy scalars exactly when one of its inputs was, which changes both how `sum` adds them and the type that comes back; the shim mirrors both.
- `max(a, b)` keeps `a` unless `b > a`.
- Truthiness is reproduced: `None` and `0` are false, NaN is true.
- Statement values pass through a strict converter, so a non-number still raises `TypeError` as the original arithmetic did.

**Parity:** random statements and profiles drive every ported function against the module as it was before Phase 3b. The statements are multi-year with NaN gaps, FCF reported or derived, with dividends and debt; the profiles have missing fields, NaN beta and analyst estimates. Results are bitwise and type-for-type equal. 65–83% of cases produce a valuation rather than an early error, and more than half pass NumPy-scalar arguments.

**Real data:** complete intrinsic-value results for AAPL, KO, JPM, NVDA, MSFT, XOM, O, BRK-B, BAC, TRV, PLD and DUK are identical to the pre-3b module, with zero differences. That set covers operating companies, banks, an insurer, REITs and a utility, with 10–12 of 12 models valued each. Full suite: 1,193 tests pass.

## Phase 4a (25 Oct 2026): Buffett quality metrics

**What moved** (`buffett.rs`):
- the generic, bank and REIT metric models (`_compute_*_metrics`) in full;
- `_total_debt` and `_cagr`, which stay as Python wrappers over Rust so their unit tests exercise the running code;
- the dead helpers `_aligned`, `_common_periods`, `_owner_earnings`, `_stdev` and `_median` are deleted.
- What stays Python: `compute_metrics`, meaning the EDGAR loads, the split-consistent share series, the ratio bridge through `financial_ratios`, and coverage.

**Python semantics reproduced:**
- **Raw values flow through.** Several metrics read values without `_finite`: the revenue deltas (`current or 0`), loss-year counts (NaN counts as observed), and share counts (infinity passes `v > 0` and is filtered later by `_cagr`). So concept values cross with `None`, NaN and infinity intact.
- **Counts stay Python ints,** and keys keep insertion order, the conditional ones included.
- **`statistics.pstdev` is exact rational arithmetic.** The port computes the population variance as an exact big-integer fraction and takes CPython's correctly rounded `_float_sqrt_of_frac` (round-to-odd isqrt) of it. It is bit-identical on 5,000 random sets from 1e-300 to 1e300.
- `statistics.median` is the middle element or `(a + b) / 2`, on a stable sort.

**Parity**
- Fuzz: 1,500 random companies through each of the three models are equal in keys, order, types and bits, with 66–78% of metric values populated.
- Real data: `compute_metrics` on 618 filers, an even sample of the 5,561-filer universe (544 generic, 40 banks, 16 insurers, 18 REITs), has zero differences in any field.

**Speed:** about 94 µs → 80 µs per company (1.2×). The cost at this size is moving the nested dicts across the boundary, not the arithmetic. The per-company EDGAR load dominates either way.

## Phase 4b (25 Oct 2026): cross-sectional scoring

**What moved** (`rank.rs`):
- `_winsorised_percentile`: quantiles, clip and rank;
- `_pillar_scores`' row means;
- `_quality_score`'s weight-renormalised blend, which `score_value` reuses.

**What stays pandas:**
- the per-model `groupby`, the index and `metrics_to_frame`;
- the final sort and rank in `combine`/`rerank`. pandas' default sort is NumPy's unstable quicksort, whose tie order is its own;
- the one-expression blends in `combine` and `blend_scores`;
- `compute_market_metrics`, which is EDGAR reads and one-line formulas.

**pandas semantics reproduced:**
- `Series.quantile` is NumPy's linear percentile at `q * 100.0` over the non-NaN values, so an infinite value can make a bound infinite or NaN.
- `clip` ignores a NaN bound and swaps reversed bounds. Its second step compares the *original* value against the upper bound.
- `rank(pct=True)` averages each tie group as an integer sum of positions over the group size, then divides by the non-NaN count.
- Row means skip NaN and add left to right.

**Parity:** in random groups (NaN, infinities, heavy ties, constant columns, groups of 1–4, object and integer columns, all four model specs), the percentile, pillar scores, quality score, `score_quality` and `score_value` match bitwise, with the same index, name and dtype.

**Real data:**
- `score_value` on the latest stored run (1,221 ranked companies: 799 generic, 246 banks, 111 REITs, 65 insurers) is identical.
- `score_quality` on all 5,561 filers in the universe (metrics and gates recomputed, 1,225 eligible) is identical in every column.
- Scoring speed: 9 → 3 ms for value, 93 → 45 ms for quality. A full metrics pass over the universe takes 75 s, nearly all of it EDGAR loads.
- Full suite: 1,213 tests pass.

## Phase 5a (25 Oct 2026): projections

**What moved** (`projection.rs`):
- `compute_projection`'s fit and cone: log returns, moments, drift shrinkage, and the parameter-uncertainty spread `σ²·t·(1 + t/n)`;
- the whole of `walk_forward_errors`;
- `summarize_errors`' statistics.
- The dead helpers `_future_position`, `_predictive_moments` and `_norm_cdf` are deleted.
- What stays Python: preparing the returns (pandas `pct_change`/`clip`), and `_build_replay` and `backtest_projection`, which are date arithmetic and orchestration.

**Semantics measured:**
- `np.log1p`, `math.log1p` and `math.erf` are the C library's here. `math.erf` has no Rust equivalent, so the port calls the C library's `erf` directly.
- Python's `round` ties to even: ten years is `round(3652.5)` = 3,652 days.

**Speed:** the walk-forward originally refitted the projection on a growing window at every monthly anchor, re-deriving that window's returns through pandas each time, which is quadratic in the history. In Rust each window's returns are a prefix of one array. The moments are still recomputed per window from scratch, so every number is the original's. `backtest_projection` on 20 years of daily closes drops from about 150 ms to about 7 ms, 20× faster.

**Parity:**
- Random wealth series (calendar-daily, business-daily and weekly dates, a plain integer index, gaps, crashes past the clip, partial benchmarks) match bitwise.
- On the real 2005–2026 histories of SPY, AAPL, KO, BRK-B, MSFT and XOM, with SPY as the prior, the projection and the full backtest (four horizons plus the replay) are identical.

## Phase 5b (25 Oct 2026): valuation bands, trend signal, stress response

**What moved** (`records.rs`):
- `valuation_history.bands`' statistics: median, quartiles, range, and today's percentile;
- `strategies.evaluate_trend_signal`'s averages: the active and provisional SMAs, the flip price and the per-month history;
- `track_record.stress_response`'s peak, trough and recovery search.
- What stays Python: the EDGAR point-in-time reads, the price loads, `_apply_sector_cap`'s selection over a pandas sort, position sizing (one expression per position), and the result dictionaries.

**Found during parity:**
- **Iterating a pandas Series yields Python floats, not NumPy scalars,** so `sum(list(series))` takes CPython's compensated path. The provisional SMA was 1 ulp off until that was corrected.
- **A one-month average's `iloc[-(1 - 1):]` is `iloc[0:]`, the whole history.** The port reproduces it.

**Parity:**
- Fuzz: random populations, month-end histories (window lengths 1–12) and annual series with gaps, losses, NaN and ties match bitwise.
- Real data: bands and stress response for 253 real companies are identical (86 had enough history for bands), as is the trend signal on SPY, QQQ, IWM and EFA at 1-, 10- and 12-month windows.

## Phase 6a (25 Oct 2026): FIFO lots and realized gains

**What moved** (`fifo.rs`): the lot book behind `portfolio_analyzer.calculate_fifo_lots_and_gains`. That covers transfers that move lots and re-sort the destination, global splits, both spin-off legs, buys that open lots, and FIFO sells with their local and display cost bases.

**What stays Python:**
- Reading each row with the original's quirks. A missing destination reads as `"NAN"`; a currency that is not three letters falls back to the default.
- The skip rules and the split de-duplication.
- The FX lookups and the display-gain fallback.
- Building the gains frame and the lot dicts. Each lot's date, FX and id are the original objects, reattached by the event that opened the lot.

**Two changes to how the work is done** (the results are unchanged):
- The shim reads the columns once instead of building a row Series per transaction with `iterrows`.
- It asks the historical FX lookup once per (currency, day). kitmatan's book makes 1,495 lookups, of which only 326 are distinct cross-currency ones, and that lookup was 87% of the function's time.

**Reproduced:**
- `holdings_long` is a `defaultdict`, so reading a key creates it. The returned dict keeps the creation order, holdings left empty included.
- Python's `min` tie rule.
- Transfers drop a lot only when it is *at or below* the tolerance, while sells drop it only *below*.
- The destination sort is stable by (purchase date, id); ids are compared by rank.

**Parity:**
- Fuzz: 1,500 random ledgers (about 2,000 sales and 3,400 lots, plus 460 holdings left empty) match bitwise, type for type. The gains frame's columns, dtypes and index order match too. The ledgers include NaN, zero and negative quantities, missing prices, all four kinds of FX answer (float, NumPy float, NaN, None), repeated splits, and same-day lots with shuffled ids.
- Real data: kitmatan (5,024 rows, 207 sales) and dheematan (3,914 rows, 210 sales), in USD and THB. The FIFO output and the whole portfolio summary are identical; the only difference is the `calculated_at` timestamp.
- **Speed:** 150–180 ms per call against 340–510 ms for the original.

## Phase 6b–c (25 Oct 2026): daily valuation and TWR accumulation

**What moved** (`history.rs`):
- **6b, the valuation step of `_value_daily_holdings_vectorized`.** Each (day, symbol, account) cell is valued at qty × price × FX, using the last trade price where the market has none. Empty and excluded cells are zero. Cash is added and each day is summed.
  - The original built four temporaries the size of the whole grid. On kitmatan that grid is 8,883 days × 89 symbols × 18 accounts, so 14M cells and 115 MB per array.
  - It also ran `np.unique` over a 14M-element mask just to feed a debug log. The log now reads the (day, symbol) price matrix instead, and only when debug logging is on.
- **6c, the daily scans of `_calculate_accumulated_gains_and_resample`:**
  - the time-weighted gain, with its bridge over −100% days;
  - each benchmark's accumulated gain;
  - the cumulative flow, absolute gain and ROI;
  - the drawdown.
  - Two debug f-strings rendered `tail().to_string()` on every call even with debug off; they are now guarded.

**What stays Python:**
- Aligning prices and FX onto the grid: `reindex`, forward fill, interpolation, and the pair and orientation rules.
- The fund-NAV and manual-override fallbacks.
- Resampling, the 16:00 New York index shift, and the range filter with its t−1 baseline normalisation.
- The requested period's `prod()` of the gain factors.

**Reproduced:**
- `np.sum(V, axis=(1, 2))` sums each day's contiguous (symbol, account) block as a single run, buffered in 8,192 elements like a 1-D sum.
- pandas `cumsum`/`cumprod`/`cummax` are NumPy accumulations that start from the first element; `cummax` skips NaN and prefers 0.0 over −0.0.
- A benchmark with duplicate dates fails the whole calculation exactly when the original's `reindex` refuses it, which is not when the indexes are identical. The shim runs the same pandas expression to decide.

**Parity:**
- 6b fuzz: 360 random grids match bitwise (series, error flag, status). They include market feeds in each price column, published NAVs, manual overrides, gaps with and without last-trade fallbacks, NaN and negative quantities, excluded accounts, missing FX pairs, and three grids wider than one reduction buffer per day.
- 6c fuzz: 480 random daily frames match bitwise (frame, TWR factor, status) across D, 1d, W, M, ME and an unknown interval. They include NaN, −100%, −0.0 and doubling days, gappy, empty and missing benchmarks, negative cost bases and random ranges.
- Real data: the recorded valuation and accumulation calls for kitmatan and dheematan in USD and THB are identical. So is the accumulation over D/W/M/ME × three ranges (48 combinations).
  - The whole `/history` result (6,119 points for kitmatan) is identical with each function swapped on its own and with both swapped at once.
- **Speed:**
  - Valuation: 58–75 ms against 173–201 ms.
  - Accumulation: 11 ms against 14 ms. The arithmetic was always under 1 ms; the gain is the debug guards.

## Final benchmark (25 Oct 2026): the whole port against the Python it replaced

`scripts/benchmark_rust_full.py`, on real data, with market data recorded once and replayed so only computation is timed. Each row is an interleaved median, Rust against the frozen original in `tests/reference/`. The three replay kernels are compared with warm Numba, which they replaced. "End to end" swaps every ported function at once: the kernels, IRR, FIFO, the valuation, the accumulation and the risk metrics. Raw numbers are in `profiles/benchmark_rust_full.json` (git-ignored).

**Per portfolio** (kitmatan / dheematan; full history of 8,883 days, 89–94 symbols, 18 accounts):

| Area | Rust | Original | Speed-up |
|---|---|---|---|
| Daily holdings replay (8,885 days × 5,024 / 3,914 tx) | 20.5 / 20.6 ms | 26.5 / 26.2 ms (Numba) | 1.3× |
| Analyzer replay | 73 / 58 µs | 121 / 83 µs (Numba) | 1.4–1.7× |
| Single-date holdings | 40 / 39 µs | 35 / 37 µs (Numba) | 0.9× |
| IRR, every holding in one summary (144 / 149) | 2.6 / 2.9 ms | 128 / 260 ms | 49–89× |
| Risk metrics, 8,883-day series | 0.47 / 0.42 ms | 1.4 / 0.96 ms | 2.3–2.9× |
| FIFO lots and realized gains | 158 / 169 ms | 334 / 330 ms | 2.0–2.1× |
| Daily holdings valuation | 54 / 68 ms | 182 / 165 ms | 2.4–3.4× |
| TWR accumulation | 7.3 / 7.7 ms | 9.6 / 10.0 ms | 1.3× |
| Projection backtest | 8.1 / 8.2 ms | 167 / 170 ms | 21× |
| Projection cone | 0.52 / 0.53 ms | 0.59 / 0.60 ms | 1.1× |
| **`/summary` computation** | **1.68 / 1.60 s** | **1.96 / 2.03 s** | **1.2–1.3×** |
| **`/history` computation, full history** | **1.01 / 1.14 s** | **1.12 / 1.32 s** | **1.1–1.2×** |
| **First call in a fresh process** | **47 / 48 ms** | **4.9 / 5.4 s (JIT)** | **~110×** |

**Market-wide:**

| Area | Rust | Original | Speed-up |
|---|---|---|---|
| Intrinsic value, 12 companies, all models | 1.74 s | 1.75 s | 1.0× |
| Buffett metrics arithmetic, 306 filers | 36 ms | 38 ms | 1.1× |
| Ranking: value scoring, 1,221 companies | 1.4 ms | 6.6 ms | 4.7× |
| Ranking: quality scoring, 5,561 filers | 43 ms | 80 ms | 1.9× |
| Valuation bands, 60 companies | 1.00 s | 1.02 s | 1.0× |
| Track record, 60 companies | 2.98 s | 2.95 s | 1.0× |
| Trend signal, SPY | 0.53 ms | 1.0 ms | 2.0× |

**What this says:**
- **Requests got 10–25% faster:** about 0.3–0.4 s per summary and 0.1–0.2 s per full history. The summary saving is mostly IRR and FIFO; the history saving is the valuation grid.
  - Much of the summary's remaining 1.6 s is the pandas and data plumbing the port deliberately left in Python.
- **The big multiples are on what was slow Python arithmetic:** IRR (50–90×), the projection backtest (21×), the value scoring (4.7×). They are small in absolute terms per request.
- **Cold start is the largest single win.** The 5-second Numba JIT compile on every edit, restart or deploy is now under 50 ms.
- **The per-company views (intrinsic value, bands, track record, Buffett metrics) are unchanged.** Their time is EDGAR and statement loading, which stayed Python.
- **Against Numba the kernels are 1.3–1.7× faster,** except the single-date holdings kernel. It is about 10% slower, but at 40 µs and off the request path.
- **Nothing changed in the numbers:** every area is bitwise identical to the original on fuzzed and real inputs.

## Phase 7a (25 Oct 2026): historical FX lookups and the IRR/MWR cash flows

The final benchmark's profile put most of a summary's remaining time here, in pandas called once per row or per holding rather than in arithmetic.

**The historical lookup** (`asof.rs`): `get_historical_price` made a pandas `asof` call for each of about 1,300 lookups per summary. Each price frame now gets an `investa_core.PriceTable`, built once and found again by the frame's identity, which is checked by weak reference, length, columns and first and last dates.
- A plain-date lookup becomes a binary search.
- The table reproduces `DataFrame.asof` exactly: it takes the last row at or before the day with no NaN in *any* column, and NaN when there is none. For a day before the first row, the original backfills with that row's price even if another column of the row is NaN, and the table does the same.
- What the table doesn't cover keeps the original pandas path: indexes in another time zone, datetime targets, dates outside the nanosecond range, and frames without a numeric price column. An index stored in s/ms/us is converted to nanoseconds first.
- Price frames are treated as immutable once built, as the market-data cache already shares them between requests.

**The cash flows** (`cashflows.rs`): `get_cash_flows_for_symbol_account` re-filtered the whole ledger for every holding. Each call ran an equality mask, a case-insensitive regex over every Note (780,000 searches per summary) and an `iterrows` pass. `get_cash_flows_for_mwr` ran `iterrows` with four `pd.to_numeric` calls per row.
- A `CashFlowLedger` now reads the frame once. It groups rows by symbol, builds each account's mask once with the original's own pandas expressions, and parses each row once. `_build_summary_rows` shares one ledger across all its holdings.
- Every row's flow is computed in Rust, branch for branch, for both builders.
- Python keeps the row selection and sort, each transfer's direction (decided from account names and Notes), the MWR's FX conversion and the result assembly.
- A text cell that `pd.to_numeric` turns into a NumPy scalar makes the original's MWR flow a NumPy float, and the IRR solver treats NumPy floats differently. So Rust reports which inputs each flow used, and the shim gives the flow that type. The holding builder casts every flow to `float` anyway.

**Parity:**
- Lookup fuzz: 1,500 random frames, each asked 21 questions, match bitwise and type for type. The frames are naive, UTC or New York indexed, in four time units, sorted, unsorted or with duplicate dates, with rows missing a column, NaN prices and integer prices, and each price-column name. A frame that grows in place gets a new table.
- Cash-flow fuzz:
  - 480 random ledgers × 12 holdings, through both the one-shot function and a shared ledger, plus 720 MWR cases under both flow bases, with and without a scope, with historical and current FX.
  - All match bitwise and type for type, with the 410 failures matched exception for exception.
  - The ledgers include regex-like account names, transfers named only in the Note, a shortable symbol, every row type, NaN, zero and negative amounts, same-day rows with shuffled ids, and integer and text-typed numeric columns.
- Real data: the full summary for kitmatan, dheematan and atchamatan, in USD and THB, is identical. So is the full history for kitmatan and dheematan.
- **Speed:** a summary takes 0.49–0.58 s against 1.33–1.67 s on the two large books, and 0.14–0.25 s against 0.34–0.74 s on atchamatan's. That is about 3× faster.

## Phase 7b (25 Oct 2026): the ratio history and the derived statement rows

These dominated the ranking batch, about 27 ms per company, and they are the same code the per-stock pages run.

**The ratio history** (`ratios.rs`): `calculate_key_ratios_timeseries` read statements one cell at a time through `_get_statement_value`, about 330 pandas `.loc` reads per company. It also re-parsed every column's date to find each period's column, which is quadratic in the number of periods.
- Each frame now resolves a line item once across all its periods: the label's value, else the first alias that has one.
- Each column's date is parsed once.
- About 48 lookups per period feed Rust, which does the arithmetic. That covers margins, returns, liquidity, solvency, ROIC, free-cash-flow margin, growth and the multiples at the period's price.
- Python's truthiness is reproduced exactly. `a or b` returns `a` only when it is a non-zero number, so a filed 0.0 falls through to the next label. The level figures and the price stay `None` where the original stores None, so an all-missing column keeps its object dtype.
- Resolving lookups up front equals reading them in the original's order only when no read can fail. Frames with duplicate labels or columns, or with non-numeric cells, keep the original code, retained verbatim as `_key_ratios_python`.

**The derived rows** (`edgar_provider._add_derived_rows`) are Total Debt, EBIT, Free Cash Flow and Total Liabilities. There is no arithmetic worth porting here; the cost was one `.loc` write per period, each of which could grow the frame.
- Each line is now written once, with the same last-write-wins order and the same new row at the end.
- A company's ISO period ends are parsed together. For `YYYY-MM-DD` strings the vectorised parse gives exactly the single-string result, NaT and time unit included.
- Frames that aren't unique float64 keep the cell-by-cell write.

**Parity:**
- Fuzz: 900 random statement sets (5,139 periods) match bitwise and type for type, frame dtypes and index included. They cover every label and alias read, filed zeros, negatives, NaN, integers, object columns with None, partly overlapping periods, a missing or empty cash flow, annual and quarterly spacing, and prices as a frame, a series or none, some too stale to count. 60 frames with text cells or duplicate labels take the original code and match.
- Derived rows fuzz: 800 random concept sets produce identical frames. They include invalid and out-of-range period ends, labels already present, and object frames.
- Real data:
  - Annual statements and ratio histories for all 5,561 filers (49,907 periods) are identical, as are the quarterly ones for the 33 filers that have them.
  - For AAPL, MSFT, KO, JPM, O, BRK-B, XOM, NVDA, PG, META, COST and V, the stock-detail path is identical three ways: annual with prices, quarterly trailing-twelve-month with prices, and the ranking's unpriced call. No real frame needed the fallback.
- **Speed:**
  - Over the universe, statements take 31.7 s against 46.1 s, and the ratio history 5.8 s against 16.7 s.
  - The ranking's full metrics pass (`collect_metrics`, 5,561 filers) takes **45.8 s against 81 s**.
  - What remains is mostly SQLite: a new connection per concept read (about 2,900 per 279 filers) and the reads themselves, plus pandas frame construction.

## Phase 8 (25 Oct 2026): the price read, the calendar, the EDGAR reads

A profile after Phase 7 found the arithmetic almost all in Rust. What was left was I/O and pandas:
- On `/history` (1.43 s for dheematan), 0.8 s went on reading prices out of SQLite and parsing their dates, against 0.04 s in the Rust kernels.
- On the ranking, the time went on about ten EDGAR reads per company, each with its own connection.
- On every request, the NYSE calendar was rebuilt per call.

One item is a Rust port; the other three were cheaper to fix in Python.

**The daily price read** (`sqlite_read.rs`, behind `market_db.get_ohlcv_batch` and `get_ohlcv`):
- **The cost:** `pd.read_sql_query` built a Python tuple per row and an object per cell (about 340k rows × 8 for a full history), then `pd.to_datetime` parsed every date string.
- **The port:** Rust steps the rows in C and writes column arrays; Python builds each symbol's frame from slices of them.
- **One SQLite in the process.** Two copies of SQLite in one process can corrupt a database through POSIX locks. So the crate links no SQLite: `sqlite_bind` is handed the path of Python's `_sqlite3` extension, opens that already-loaded image with `RTLD_NOLOAD`, and resolves the C API through it. It is the same library Python calls, on macOS, in Docker and in CI. If it can't be resolved, the reader stays off and pandas reads.
- **dtypes as pandas infers them over the whole result:** int64 when every cell is an INTEGER, float64 (NULL as NaN) otherwise. An all-NULL column is float64 NaN, which is what the original's `to_numeric` makes of the object column `read_sql_query` returns.
- **Declines to the original:**
  - a date that isn't `YYYY-MM-DD` inside the datetime64[ns] range (the original then raises or parses exactly as before);
  - text or a blob in a price;
  - a non-UTF-8 symbol;
  - a symbols argument that isn't a list of `str`;
  - a database it can't open;
  - an empty single-symbol read, whose frame keeps the SQL column names.
- **Python fix on the same path:** `market_data.get_historical_data` then made each index UTC with `pd.to_datetime(index, utc=True)`, about 3.4 ms per symbol. For a naive DatetimeIndex, `index.tz_localize("UTC")` gives the identical index in 6 µs. That was checked with NaT, ns/us/s units, a business-day frequency, intraday stamps and duplicate dates.

**The NYSE calendar** (`utils_time.nyse_schedule`): every `cal.schedule()` call rebuilt the holiday table, about 30 ms warm and 130 ms cold, and a summary made several of them.
- One schedule is now built per process, from 1970 to five years ahead (0.28 s, once), and each call is a 0.1 ms slice of it.
- A range with no session returns the calendar's own empty frame, whose columns are object rather than UTC stamps.
- Anything else goes to the calendar itself: a reversed range (which raises), a datetime, Timestamp or string, or a day outside the window.
- The five direct call sites and `_cached_schedule` use it.

**EDGAR read sessions** (`edgar_provider.company_session`): scoring one company read its facts about ten times, each through a new connection. Those reads were the whole concept set, the statement subset, then one concept at a time for every split-sensitive series.
- Inside a session the reads share one connection. The assembled-series and by-filing readers query each tag once (the first read prefetches every chain tag) and answer later subsets from what is loaded.
- That is exact because a tag's rows don't depend on which other tags share the query. That was checked on all 50 tags that hold same-day ties between two filings, against 350 random tag sets and the full set.
- The cache dies with the block. A session is per company, per as-of date and per thread, and a context copied into another thread reads without it.
- Callers get copies, so mutating an answer can't change a later one.
- `compute_metrics`, `get_statements` and each company in `collect_metrics` open one.
- Valuation bands do not: they read a few concepts once, and the prefetch made them slower (0.8 s → 1.4 s for 60 companies).

**Parity:**
- **Price read, fuzz:** 48 random archives with 672 requests match the frozen originals (`tests/reference/ohlcv_read.py`) bitwise and type for type: dict order, columns, dtypes, index and every cell, exceptions included.
  - The archives carry NULL prices and volumes, all-NULL columns, integral prices, fractional volumes, text prices, impossible, misshapen and out-of-range dates, odd symbols, another interval, price bases and corporate actions.
  - 446 reads went through Rust, 909 were empty and 108 declined.
- **Price read, real data:** all 5,670 symbols in the archive (20,708,291 rows) read identically, in 9.5 s against 33.1 s. Single-symbol reads match for 100 symbols under all three adjustments.
- **Calendar:** 360 random ranges plus the edges, holidays, weekends, reversed ranges and non-date arguments give the same frames as `cal.schedule()`.
- **EDGAR sessions:** `collect_metrics` over all 5,561 filers, `get_statements` for all of them, bands and track records for 60 companies, and a 2016 as-of run are identical with sessions on and off. A unit test covers restatements, same-day ties, NULLs, subsets, nesting, threads and mutation.
- **End to end against the last commit (da400ec7):** the summary for all three books in USD and THB and both history windows are identical. That covers every change above, including the UTC index. Both sides ran with `PYTHONHASHSEED=0`: at the time the summary's account rows came out in a different order in every process, on the old code as on the new (since fixed, see below).

**Speed**, computation with market data replayed (median of five, last commit → now):

| | kitmatan | dheematan | atchamatan |
|---|---|---|---|
| `/history`, full | 1.06 → **0.48 s** | 1.17 → **0.49 s** | 0.71 → **0.28 s** |
| `/history`, default window | 1.01 → 0.51 s | 1.14 → 0.50 s | 0.71 → 0.27 s |
| `/summary` USD / THB | 0.59 / 0.57 → 0.48 / 0.46 s | 0.48 / 0.51 → 0.44 / 0.46 s | 0.13 / 0.25 → 0.13 / 0.19 s |

| Batch | Before | After |
|---|---|---|
| Ranking metrics, 5,561 filers (`collect_metrics`) | 47.0 s | **29.6 s** |
| Statements, 5,561 filers | 28.2 s | 20.0 s |
| Whole price archive read | 33.1 s | 9.5 s |

## Phase 9 (25 Oct 2026): the summary's per-holding filters

After Phase 8 the summary's remaining cost was pandas filtering, not arithmetic. Python fixes only; no Rust.

- **Same-day flows and last-trade price** (`_build_summary_rows`). For every holding the summary masked the whole ledger by symbol, account and date, and converted every row's date (`.dt.date`) each time. It did this once for the day's flows, and again for the last-trade price fallback on holdings with no market price.
  - `_HoldingRows` selects the report-date rows, and the priced rows up to the report date, once per summary. Each holding then filters only those by symbol and account. That is the same rows, in the same order, with the same labels.
  - When nothing traded on the report date, a holding skips the filter entirely.
  - A failure in the shared step is kept and raised again inside each holding's own handler, so the same messages are logged.
- **Per-account MWR** (`_calculate_aggregate_metrics`): the normalised `Account` and `To Account` columns (`astype(str).str.strip().str.upper()`) were rebuilt for every account. They are now built once, and a failure is replayed the same way.
- **`CashFlowLedger`:** the `Type == "transfer"` mask, which every account's Note match used, is built once, and `original_index` is read into a list once instead of once per holding.

**Parity:**
- **Fuzz:** 72 random books through the real holdings pass match the frozen originals (`tests/reference/summary_rows.py`) bitwise and type for type: rows, account currencies, flags, account-level and overall metrics, and every warning and error logged.
  - The books cover holdings without a market price, trades and transfers on the report date, days with none, text and zero prices, a text `Date` column (so the shared filters fail and every holding's handler must fail as before), no `To Account` column, and an account filter.
  - Across them the report-date filter returned rows 576 times, nothing 205 times and failed 174 times; the priced filter returned rows 380 times and failed 76 times.
- **Ledger:** the existing parity test against the original per-holding function passes unchanged.
- **Real data, against the previous commit (eee1987e):** the summary for all three books in USD and THB and both history windows are identical, in four old/new pairings (`PYTHONHASHSEED=0`).

**Speed**, computation with market data replayed (median of ten over two alternating rounds):

| `/summary` | Before | After | |
|---|---|---|---|
| kitmatan USD / THB | 0.507 / 0.489 s | **0.288 / 0.261 s** | 43–47% faster |
| dheematan USD / THB | 0.427 / 0.435 s | **0.222 / 0.222 s** | 48–49% faster |
| atchamatan USD / THB | 0.135 / 0.192 s | **0.082 / 0.122 s** | 36–39% faster |

History is unchanged (within ±3%), as expected: it does not run these functions.

**What remains** (dheematan, profiled): the account and overall totals (0.10 s, MWR flows and `safe_sum`), the dividend history (0.10 s, an `iterrows` pass over about 1,200 dividends) and the FIFO setup (0.10 s, about 15,600 single-value `pd.to_numeric` calls). All three are per-row Python work.

## Phases 10–16: the second port (25 Oct 2026 onward)

On 25 Oct 2026 the user widened the scope: every module that *can* move to Rust does, whether or not it speeds the app up. That covers calculations, storage and import. Python keeps the FastAPI routes, the network fetchers (yfinance and the HTTP clients), PDF parsing (pdfplumber), the market calendar (pandas_market_calendars) and Gemini. Two kinds of code still stay in Python: pandas frame construction, because the routes take DataFrames, and BLAS/LAPACK-backed NumPy calls, which can't be reproduced bitwise. The method is unchanged: freeze the original, fuzz for bitwise parity, then check real data.

## Phase 10 (25 Oct 2026): SQL in Rust on Python's own connections

The storage modules keep their `sqlite3.Connection` objects: connection caching, PRAGMAs and the callers' `with conn:` blocks are untouched. `rust_sql.execute`, `executemany` and `read_frame` run the statement in Rust (`sql.rs`) on the `sqlite3*` the Python connection holds. So a Rust statement joins the caller's open transaction, sees its uncommitted writes, and commits or rolls back with it.

**The handle.** In CPython 3.12 a `pysqlite_Connection` is `PyObject_HEAD` followed by `sqlite3 *db`.
- `rust_sql` reads the pointer at `object.__basicsize__`, and once per process checks the layout: a probe connection on a temporary file must report the same file through `sqlite3_db_filename` as through `PRAGMA database_list`.
- The pointer is read fresh on every call, because a closed connection sets it to NULL.
- Connections don't take weak references, so nothing is cached per connection.

**What Python's `sqlite3` does around a statement** (CPython 3.12 `cursor.c` and `statement.c`), reproduced:
- Python's own thread and closed-connection checks run first (via `conn.cursor()`, which raises exactly what `execute` raises).
- Under legacy transaction control, DML in autocommit runs `"BEGIN " + isolation_level` first, before binding, so a binding error leaves the transaction open as it does in Python. DML means `lstrip_sql` (whitespace, `--` and `/* */` comments) followed by INSERT, UPDATE, DELETE or REPLACE.
- Only one statement is allowed per call.
- The binding count must match, with Python's message.
- `rowcount` is the DML changes or -1. `lastrowid` is the connection's last insert rowid after `execute`, and None after `executemany`.
- Each SQLite error is raised as the class `get_exception_class` picks, with SQLite's message and the extended `sqlite_errorcode` / `sqlite_errorname`.
- Text that isn't UTF-8 raises Python's `Could not decode to UTF-8 column` error.
- `read_frame` builds the frame with pandas' own `_wrap_result`. A failure rolls the connection back and raises pandas' `DatabaseError`, as `read_sql_query` does.

**The connection itself is driven from Rust** (`pyconn.rs`, `PyConn`). It makes the checks above, binds positional and named parameters, runs the statement, and raises each error as Python would; `rust_sql.execute` / `executemany` are thin wrappers over it, and the storage ports use it directly.
- Named binding follows CPython 3.12 `bind_parameters`: each placeholder's name, less its first character, is looked up in the dict (extra keys ignored). The errors are Python's: `Binding N has no name, but you supplied a dictionary (which has only names).` and `You did not supply a value for binding parameter :name.`
- A named placeholder bound by position goes to Python, which emits its DeprecationWarning there.
- `executemany` converts every item before running any, so an item Rust can't bind sends the whole call to Python while nothing has happened yet. A binding error at item *k* is raised after items 0..*k*−1 have run, as in Python. One gap remains: a generator that raises partway is materialised before the implicit BEGIN, where Python would raise after it.

**Left to Python's own call, before anything runs:**
- a parameter Python would adapt or refuse: an adapter registered on a base type, a date, a NumPy integer, an int beyond 64 bits, a surrogate string, a `bytearray`, a float subclass with `__conform__`;
- named parameters;
- a `row_factory` or a custom `text_factory`;
- explicit `autocommit` control;
- a NUL in the query or an empty statement.

**Parity:** 32 random sequences of 120 operations (named and positional parameters, `executemany` runs that fail partway), on twin databases under every isolation level including autocommit, agree after every step. They compare each outcome (rows, description, rowcount, lastrowid, or the exception's class, message, code and name), `in_transaction`, `total_changes` and every table's contents. Across them, 1,710 statements ran in Rust, 185 raised SQLite errors and 235 programming errors (all matching Python's), and 899 were declined to Python by their parameters. Tests also cover a Rust write rolled back by the caller's `with conn:` block, the thread and closed checks, and the connections Rust must leave alone.

## Phase 11a (25 Oct 2026): `db_utils`

`db_utils.py` went from 2,234 lines to 517. Everything except `migrate_csv_to_db` (Phase 12, with the import code), the `NpEncoder` class and the module constants is now Rust (`db_portfolio.rs`, `db_screener.rs`):
- the transaction writes and loader;
- watchlists;
- the screener cache and its AI and intrinsic-value updates;
- the v1–v15 schema migrations;
- `get_db_connection` with its per-thread cache and retries;
- paths and the read-only connection.

**How the port is written.** It is PyO3 code working on the same Python objects: routes pass dicts, the loader returns a DataFrame.
- **Native Rust:** the SQL statements and the arithmetic.
- **The same Python calls,** wherever the original relied on a Python operation's exact behaviour: `str.title`, `strptime`, `pd.isna`, `str()` of a dict in a log line, `json.dumps(cls=NpEncoder)`, `sum()` (compensated for floats since 3.12), `list.sort(key=...)`, `datetime.now()`, logging with the caught exception as `exc_info`, and pandas' column operations in the loader.
- **Module lookups:** sibling functions and module state (`_DB_CONN_CACHE`, `_SCREENER_SCHEMA_INITED`, `FAVORITES_WATCHLIST_NAME`) are read from the `db_utils` module at call time, so a patched attribute is the one used, as a Python global lookup finds it.
- **Evaluation order:** kept wherever it decides which handler catches a failure. Values built before a `try` raise past it; values built inside are caught.

**Found while porting: pandas counts references.** `DataFrame.__setitem__` warns of chained assignment when `sys.getrefcount(self) <= 3`. A Python `df[col] = v` holds the frame on the interpreter's stack, one reference more than a call from Rust, so the first port warned seven times per load where the original warned never. `pyhelp::setitem` holds that reference too. The parity tests now compare warnings as well as logs.

**Parity** (against `tests/reference/db_utils.py`, frozen before the port), comparing return values, every table's schema and rows, every log record (level, message, exception type) and every warning:
- 900 random calls on twin portfolio databases (adds, updates, deletes, watchlists, loads);
- migrations from 40 random older states (any recorded version, columns dropped, tables missing, a legacy `watchlist` table to move, no `schema_version` table);
- connections on plain and cloud-drive paths, through the cache, a closed cached connection, `use_cache=False`, `check_same_thread=False`, an unwritable directory and the read-only fallback;
- 360 random screener-cache operations.

**Real data:**
- The three books load identically, with their watchlists.
- The screener readers return the same 3,007 rows.
- The migration leaves copies of all five user databases identical.
- All 1,376 tests pass.

## Phase 11b (25–26 Oct 2026): `market_db`

`market_db.py` went from 1,528 lines to 611. Every `MarketDatabase` method except `__init__` and `_get_connection` is now Rust (`mdb.rs`), and so are the module helpers `_iso_days`, `_rust_reader_bound` and `_read_daily_frames`. Each method takes the object as `slf`, so its connection, write lock, column cache and sibling methods are reached through it, as in the original.

- **Native:**
  - the ISO day strings (years 1–9999; NumPy's own formatting otherwise);
  - the split factors, reproducing NumPy's stable argsort, reversed cumprod, right-sided searchsorted and `allclose(1.0)`;
  - the dividend factors;
  - the read-time multiplies;
  - the value cleaning of the upserts (NaN as None, volumes truncated, an infinite or 64-bit-overflowing volume through Python's `int()` so it raises or grows as there).
- **The same pandas calls:** `to_numeric`, `groupby`, `merge`, `to_datetime`, and every column assignment through `pyhelp::setitem`.
- **`with a, b:`** enters `a` before it evaluates `b`, so the write lock is taken before the connection is fetched, as in the original.
- **Python's own messages:** tuple unpacking (`for day, rate in rows`) raises Python's exact errors, and `symbols + [...]` uses Python's `+` inside the connection's `with`.
- **Kept as NumPy fallbacks:** the NumPy originals of the two factor helpers stay in `market_db.py` (`_future_split_factors_numpy`, `_dividend_factors_numpy`), for inputs NumPy answers in its own way: a NaN day among the dates, an unsorted index (where its binary search is implementation-specific), closes that are not a float64 array.
- **Ported as found:** `get_share_counts` and `check_integrity`'s fallback still use `date.today()`, against the market-time rule.

**Parity** (against `tests/reference/market_db.py`):
- **Method sequences:** 700 random calls on twin archives, including a legacy archive without the provenance columns, agree on every return value (frames bitwise with their index), table, log record and warning.
- **Helpers:** 240 random indexes (ns/us/s, 1599–10000, NaT, tz-aware, unsorted) give the same `_iso_days`, split factors and dividend factors.
- **Real data:** the live 3.2 GB archive reads identically for every symbol at the split basis; for 250 symbols with corporate actions under all three adjustments; for single-symbol reads; and for FX, latest closes (now and as of a date), share counts, bases, actions, data quality, fund NAVs, first and last dates and sync stamps. 95 integrity checks on real frames, half of them perturbed, match. The archive holds no raw-basis symbols yet, so the raw-to-split conversion is covered by the fuzz alone.
- All 1,390 tests pass.

## Phase 11c (26 Oct 2026): `buffett_store`

Every `BuffettRankStore` method except `__init__` and `_connect`, and the helpers `_escape_like` and `_coerce`, are now Rust (`rank_store.rs`); `buffett_store.py` is 260 lines, down from 501.
- **`sqlite3.Row`.** The store's connections set `row_factory = sqlite3.Row`, which would normally send a statement to Python. These rows never leave the store, though: they are read as `dict(row)`, `row["run_id"]` or `row[0]`. So `PyConn::new_internal` runs them in Rust, and `pyconn::row_dict` reproduces the Row lookups: each name resolves to the first column matching it case-insensitively, as `Row.__getitem__` does.
- **Parity:**
  - 480 random calls on twin stores (runs started, finished and pruned at every `keep` and staleness; score frames with NumPy scalars, NaN, text, missing columns and integer labels; paged readers with LIKE wildcards in the search) agree on return values, tables, logs and warnings.
  - The live store (256 runs, 1,219 ranked rows in the latest) reads identically through every reader.

## Phase 11d (26 Oct 2026): the EDGAR fact store and the company session

`EdgarFactStore`'s readers, writes and ingest logs, `_effective_as_of`, `_session_for`, and `company_session` (now a Rust context-manager class) live in `ef_store.rs`. The session cache that `_read_tags` kept is folded into the two cached readers.
- **Order of results.** The readers' row loops fill the nested dicts the concept resolution reads in row order, so last-write-wins and insertion order are the original's.
- **What stays in Python.** `_CompanySession` stays a Python data holder, and the context variables stay where they were, so the session machinery is reached the same way.
- **Parity:**
  - 400 random calls on twin stores (restatements, same-day ties, NULLs, quarterly spans; random tag sets and as-of dates; inside, outside and nested in sessions; writes) agree on return values with their dict order, and on tables, logs and warnings.
  - On the live store, every reader for all 5,561 filers at two dates is identical.
  - The whole ranking metrics pass is identical whether it reads through the Rust store or the frozen original. It takes about 37 s either way: the cost is still SQLite and pandas.

## Phase 12 (26 Oct 2026): import

`data_loader.load_and_clean_transactions` and its helpers, `csv_utils.convert_csv_headers_to_cleaned_format`, the `io_handlers` writers and `db_utils.migrate_csv_to_db` are now Rust (`dl.rs`, `db_portfolio.rs`).
- **What stays pandas.** The CSV parse stays `pd.read_csv`, because its float parsing and type inference are pandas' own; the cleaning is the same pandas operations in the original's order.
- **Parity:**
  - 120 random CSV ledgers load identically through both versions (all seven results, logs, warnings). They carry header aliases, unknown and missing columns, every date shape including ordinals and invalid ones, numeric text, cash spellings, spin-off spellings, reserved types, and bad currencies.
  - So do database sources (populated, empty, unreachable), 20 header standardisations (compared on the file written), the export writers, and 10 CSV migrations.
  - The three real books load identically.
- All 1,415 tests pass.

## Phase 13a (26 Oct 2026): `corporate_actions` and the analyzer's standalone helpers

`corporate_actions` (the deferred-action arithmetic, IBKR spin-off parsing and leg building, split deduplication) and seven `portfolio_analyzer` helpers (`get_type_id`, the correlation matrix, scenario impact, rebalancing trades, HHI, the health score, projected cash interest) are now Rust (`ca.rs`, `pa_small.rs`).
- The arithmetic is Python's own number operations, in the original's operand order (so `100 * x` dispatches from `int`, and a NumPy scalar's reflected methods win as they do in Python). `sum`, `round`, `max`, `min` and `format` are the builtins, so results keep their types.
- The spin-off parser uses the module's compiled patterns, so the regex semantics stay Python's.
- `generate_cash_interest_events` still anchors on `date.today()`; that is flagged, not changed.
- **Parity:** 320 random rounds of the arithmetic, the parser, the leg builder and split deduplication, and 160 of the analyzer helpers, match the frozen originals type for type, with logs and warnings.

## Phase 13b (26 Oct 2026): the report extractors

`calculate_periodic_returns`, `extract_dividend_history` and `extract_realized_capital_gains_history` are now Rust (`pa_reports.rs`): the same pandas operations in the original's order, side effects included.

**Found while porting:** two ways the port was not the original, both invisible to a read of the code.
- **A name the linter thought unused.** `ruff --fix` removed `portfolio_analyzer`'s import of `USFederalHolidayCalendar` once no Python code used it, but the port reads it from the module at call time. The holiday filter then failed into its `except` branch and logged a warning, so a history starting on 1 January kept the holiday. `tests/test_rust_module_names.py` now scans the Rust sources for every module attribute they read and fails if one is missing.
- **An object kept alive longer than in Python.** `df_gains, _ = calculate_fifo_lots_and_gains(...)` discards the tuple at once. The port held it in a local, which kept the unfiltered frame alive, and pandas' SettingWithCopy check (a weak reference from the slice to its parent) then warned where the original never did. The tuple is now dropped where Python drops it.

**Parity:**
- 125 random rounds of each extractor match the frozen originals: histories on DatetimeIndex and RangeIndex across weekends and holidays; ledgers with taxes, interest, text amounts and missing FX; account and date filters on a non-datetime `Date`. That covers results, logs, warnings, and the inputs left behind.
- **Real data:** for all three books, the summaries in USD and THB (which run the dividend history), the realized gains under three account filters, and the 8,883-day periodic returns are identical.

## Phase 13c (26 Oct 2026): the holdings pass and the FIFO shim

`_process_transactions_to_holdings` and `calculate_fifo_lots_and_gains` are now Rust (`pa_holdings.rs`, `pa_fifo.rs`). With them, everything between the loaded ledger and the replay kernel or the FIFO book is Rust.
- **The holdings pass** normalises the caller's frame in place, as the original did. It deduplicates splits and encodes the columns with the same pandas operations; the price, cash and split-ratio masks over the encoded columns are native. The kernel's state is unpacked into holdings whose values stay NumPy scalars.
- **The FIFO shim** reads each row with the original's Python operations: `pd.isna`, `pd.to_numeric`, `str().upper().strip()`, and the parse `try`/`except`. Its arithmetic stays on the row's own objects, so a gains record's quantity keeps its NumPy type. The lot events go straight to `fifo::run`, with no round trip through Python lists.
- The FX bridge and the conversion fallback are looked up on `portfolio_analyzer` at call time. The bridge's import is now marked as read by Rust, and the module-name guard also follows `let m = py.import(...)` bindings.
- **Parity:**
  - 240 random rounds of the holdings pass match the frozen original: all nine outputs, the frame it leaves behind, logs and warnings.
  - So do 320 rounds of the FIFO shim on rougher ledgers: text and junk in numeric columns, NaT and string dates, None symbols, NaN ids, a missing required column, and the real USD-bridge lookup.
  - `test_rust_fifo_parity.py` still matches the pure-Python FIFO bitwise on 1,500 ledgers.
- **Real data:** for all three books, in USD and THB, the summaries, the FIFO output and the holdings pass are identical, and so is the frame the holdings pass leaves behind.
- All 1,445 tests pass.

## Phase 13d (26 Oct 2026): the summary rows and the totals

`_build_summary_rows` (with `_HoldingRows`, now a memo inside the port) and `_calculate_aggregate_metrics` are now Rust (`pa_rows.rs`, `pa_aggregate.rs`). `portfolio_analyzer.py` is down to 529 lines of wrappers and constants.
- **Every figure is Python's own arithmetic on the holding's objects**, in the original's operand order. A NumPy quantity therefore keeps the row's values NumPy scalars, `max(0, cost)` still returns the integer `0`, and `* 100` and `* 100.0` stay as the original wrote them. The intra-day flows were Python floats in the original, so they are native.
- Every log message formats its values with Python's `format()`, so a float in a DEBUG line reads as Python prints it.
- `safe_sum`, the FX, IRR and MWR helpers and the cash-flow ledger are looked up on `portfolio_analyzer` at call time, and their imports are marked as read by Rust.
- **The frozen speedup reference now borrows the frozen module's helpers.** `tests/reference/summary_rows.py` copied its globals from the live `portfolio_analyzer`. Once the live module stopped importing `defaultdict`, the reference failed with a `NameError` and the old parity test reported a mismatch.
- **Parity:**
  - 300 random rounds of the rows match the frozen original, with logs down to DEBUG and the ledger left behind. The holdings are hand-built with NumPy, int and float values, NaN costs, shorts, and cash in four currencies. The market data has text, zero and NumPy-integer prices. The rounds also cover published NAVs, valid and invalid manual prices, missing FX, interest with thresholds, and account filters.
  - So do 200 rounds of the totals: dropped and text columns, an empty frame, every account-filter shape, and ledgers with and without `To Account`.
  - The pre-speedup parity test still passes.
- **Real data:** for all three books, the summaries in USD and THB, with and without an account filter and with cash interest on, are identical with the rows and totals swapped.
- **Speed:** `/summary` is unchanged: 274 vs 283 ms and 273 vs 276 ms on the two large books.
  - A first cut of 13c was 15 ms slower in the FIFO shim. The cause was two things done per row: looking pandas functions up by a freshly built name, and calling `py.import` inside the holdings pass's per-row closure.
  - Binding `pd.isna`, `pd.notna` and `pd.to_numeric` once per call, and interning the method names, brought each ported function within 1–2 ms of the original.
- All 1,464 tests pass.

## Phase 13e (26 Oct 2026): `portfolio_logic`'s summary

`calculate_portfolio_summary` and its helpers are now Rust (`pl_summary.rs`, `pl_helpers.rs`). That covers the transactions fingerprint, the default metrics, the published NAVs, the sector and geography enrichment, and the historical FX table. With 13a–d, the whole `/summary` calculation is Rust from the loaded ledger onwards. `portfolio_logic.py` is down to 668 lines, most of them re-exports and the `__main__` block.
- **It orchestrates Python objects:** the market provider, the NAV store, pandas frames, and the FIFO cache under its lock (`mdb::with1`). So every call is Python's, in the original's order, including the sets whose iteration order depends on how they were filled.
- Module state and siblings are looked up on `portfolio_logic` at call time. That covers `_FIFO_CACHE`, `SHORTABLE_SYMBOLS`, the clock functions and the analyzer functions it imports, so the tests and benchmarks that patch them still reach the patched one.
- **One unavoidable difference.** The two `logging.error(traceback.format_exc())` lines print the caught exception's traceback, which has no frame for the summary function now that it is not Python. The exception line itself is unchanged.
- **Parity:**
  - 160 random summaries match the frozen original, each run twice so the FIFO cache is read as well as filled. They cover the whole seven-part result, the caller's frame afterwards, and the logs, with the elapsed-time line and the traceback bodies masked.
  - The runs use a deterministic fake provider: critical and partial fetch errors, both Yahoo FX conventions, a currency with no history that takes the current-quote fallback, and metadata from the batch, the per-symbol fetch, or a failing service. A fake NAV store stands in for the real one.
  - The ledgers cover every account-filter shape, transfer-source backfill, splits, "All Accounts" rows, a missing `original_index`, and manual overrides with valid, text and zero prices.
  - Empty, missing and account-less inputs, and the helpers, are compared separately.
- **Real data:** for all three books, with the real market provider, in USD and THB, with and without an account filter, the whole summary is identical. The original, run again, is stable, so the quotes did not move under the comparison.
- **Speed:** `/summary` end to end is level with the Phase-13 Python: 359 vs 375 ms (kitmatan) and 288 vs 285 ms (dheematan). atchamatan's small book is 5–8 ms slower (119 vs 111–114 ms).
- All 1,474 tests pass.

## Phase 14a (26 Oct 2026): `portfolio_cashflows`

The daily net external cash flow, over a date range (the vectorized pass the history uses) and for one day, is now Rust (`pc_flows.rs`). `portfolio_cashflows.py` is down to 80 lines.
- **It is pandas and per-row Python**, so every operation is Python's own, in the original's order.
- **A frame the original rebinds (`df_cash = df_cash[mask]`) is reassigned here, not shadowed.** A Rust shadowing `let` would keep the parent frame alive until the end of the scope. Pandas' SettingWithCopy check, a weak reference from the slice to its parent, would then warn on the next column assignment where the original never did.
- **The original's side effects are kept.** It rewrites the index of the FX and price series it is handed, and it appends to a scratch file (see "Found while porting").
- **Parity:**
  - 320 random rounds of the vectorized pass and 240 of the per-day pass match the frozen original. The comparison covers results, logs, warnings, the scratch-file lines (captured, not written), and the FX and price tables left behind.
  - The ledgers cover cash and stock deposits, transfers across and within a selection, Note-named clean-ups, flows before the range, and currencies with and without FX history, over UTC and naive ranges.
  - Unpriced transfers fall back to the price history (date or Timestamp index, sorted or not) and then to another trade.
- **Real data:** every vectorized call that the history for the three books makes is identical: USD and THB, all accounts and IBKR alone.

## Found while porting

- **The vectorized cash-flow pass writes to another tool's scratch file** (not changed). A transfer that neither the price history nor another trade can price appends `Missing price for transfer: <symbol>` to `/Users/kmatan/.gemini/antigravity/brain/19ed956f-…/scratch/missing_transfer_prices.txt`, a path left behind by a debugging session. The directory exists on this machine, so the server really does write there. The port keeps it, and the parity test captures the writes instead of making them. Removing it is a one-line change once you want it gone.
- **A short holding's "Avg Cost (Local)" is the previous long holding's cost basis divided by the short quantity** (not changed). `_build_summary_rows` assigns `cost_basis_display_local` only in the long branch but reads it for every holding with a non-zero quantity. A short row therefore divides whatever the last long holding left behind, and a book whose first non-zero holding is short raises `UnboundLocalError` out of the summary. No real book holds a short, so no one has seen it. The port keeps both behaviours, and `test_short_holding_reads_the_previous_cost_basis` pins them.
- **A holding's IRR matches its account name against transfer Notes as a regular expression, case-insensitively** (not changed). That is `str.contains(account, case=False)` in `get_cash_flows_for_symbol_account`.
  - atchamatan's `E*Trade` account reads as `E*Trade`, i.e. any number of E's followed by "trade". So a transfer of the same symbol whose Note contains "trade" anywhere, such as "TD Ameritrade", is pulled into the E*Trade holding's flows.
  - An account name with parentheses, like `IBKR (1)`, is a regex group: pandas warns about it, and `(1)` matches a bare `1`.
  - The ledger keeps the original's expression, so the behaviour is unchanged. Fixing it needs `regex=False`, a deliberate change with its own before-and-after check.
- **Fixed 25 Oct 2026: the summary's account rows came out in a different order in every server process.** Account ids were assigned in the iteration order of a set of account names, and Python seeds string hashing per process, so two runs of the same book disagreed on row order and on the holdings dict's order; the values were the same. `_process_transactions_to_holdings` now sorts the accounts by name, the order the account picker already uses. All three books' summaries are identical under different seeds and, once aligned, identical to before. `tests/test_account_order.py` runs four seeds in separate interpreters and fails without the fix.
- **`portfolio_analyzer.generate_cash_interest_events` anchors its projection on `date.today()`** (not changed), so on the Bangkok host it can start a month ahead of the US market's.
- **`market_db` reads dates off the server's clock in two places** (not changed). `get_share_counts` ages entries with `date.today()`, and `check_integrity` falls back to it if `utils_time` fails. Both predate the market-time rule. The share-count cutoff can therefore be a day ahead of the US market on the Bangkok host.
- **Known type corner, not reached by real data:** in an MWR transfer into the scope, `-(abs(qty) * price)` stays a NumPy *integer* when both cells were text holding integers, such as "3" and "5". The port makes it a NumPy float with the same value. The loader never produces text-typed numeric columns.
- **`backtest_projection` raises on a series without dates when a value series is passed** (not changed). It subtracts a `Timedelta` from an integer position. The app only ever passes a dated TWR series, so no user reaches it; the port fails the same way, and the parity test checks that.

- **Fixed 24 Oct 2026: tax rows were booked twice whenever the Auto-cash block skipped them.** That covered every Manual-mode tax row and every tax row on `$CASH`.
  - **Cause:** the original tax branch (Apr 2026) was stranded when taxes moved into the Fees/Tax branch (May 2026). Once the Auto-cash block was inserted above it, it became an `elif` of that block.
  - **Effect:** such a row added its amount to taxes a second time (falling back to Commission when Price/Share was zero), took it off dividends, and raised the historical-FX cost basis. Total gain therefore lost the tax twice.
  - **The fix:** delete the stranded block in Rust and in the reference, so every tax row books the way an Auto-mode stock tax row always did.
  - **Measured on real data:**
    - atchamatan: taxes 42,946 → 25,943 USD and total gain +17,003, so total return goes from 39.58% to 39.97%. All of it is the `$CASH` @ E*Trade position, whose dividends had shown −16,747 instead of +256.
    - dheematan: taxes −2 USD.
    - kitmatan: unchanged.
  - **Test:** `test_a_tax_row_is_booked_once` fails on three of its four cases without the fix.
- **The single-date holdings kernel is off the request path.** It sits behind `HISTORICAL_CALC_METHOD = "numba"`, while the default is `numba_chrono`.
- **The kernel functions keep their `_numba` names** because callers import them by those names. Renaming is a later cleanup.
