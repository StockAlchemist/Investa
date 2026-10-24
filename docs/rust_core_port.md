# Rust Calculation Core: Port Plan and Status

**Status:** Phase 1 done (the Numba kernels). Phases 2–6 to do.
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
| 2 | Financial math: `finutils` NPV/IRR, `risk_metrics`, `buffett_metrics` helpers | To do |
| 3 | Intrinsic-value Monte Carlo models in `financial_ratios` (random draws stay NumPy-generated, so outputs stay identical) | To do |
| 4 | Buffett ranking: `buffett_metrics`, `buffett_value`, `buffett_rank` | To do |
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
