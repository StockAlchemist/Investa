# Rust Calculation Core: Port Plan and Status

**Status:** Phases 1–3 done (the Numba kernels; IRR/NPV and risk metrics; the intrinsic-value models). Phases 4–6 to do.
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
| 4 | Buffett ranking: (a) `buffett_metrics`, **done**; (b) `buffett_value`, `buffett_rank` | In progress |
| 5 | `projections`, `valuation_history`, `strategies` math | To do |
| 6 | Pandas-heavy portfolio engine: FIFO lots and realized gains, `portfolio_cashflows`, history valuation | To do |

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
| Full history request | 4.2 s / 6.3 s | 4.1 s / 6.4 s | n/a |

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

## Found while porting

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
