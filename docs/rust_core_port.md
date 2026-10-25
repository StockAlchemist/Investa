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
| 7 | What the final benchmark left on the request path: (a) the historical FX lookup and the IRR/MWR cash flows; (b) statement assembly and the ratio history | (a) **Done**; (b) to do |

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

## Found while porting

- **A holding's IRR matches its account name against transfer Notes as a regular expression, case-insensitively** (not changed). That is `str.contains(account, case=False)` in `get_cash_flows_for_symbol_account`.
  - atchamatan's `E*Trade` account reads as `E*Trade`, i.e. any number of E's followed by "trade". So a transfer of the same symbol whose Note contains "trade" anywhere, such as "TD Ameritrade", is pulled into the E*Trade holding's flows.
  - An account name with parentheses, like `IBKR (1)`, is a regex group: pandas warns about it, and `(1)` matches a bare `1`.
  - The ledger keeps the original's expression, so the behaviour is unchanged. Fixing it needs `regex=False`, a deliberate change with its own before-and-after check.
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
