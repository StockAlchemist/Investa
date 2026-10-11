# Moving the Backend to Go or Rust: Evaluation

**Status:** **Declined.** No rewrite in either language, and no Rust module for now. The measured
bottleneck was the market-data sync, not Python, and a small Python fix to it cut a cold
`/history` from 18.6s to 10.6s.
**Written:** 11 Oct 2026
**Read this first:** the Numba kernels, which are the only part of the backend Rust could speed
up, take **under 0.1s** of a cold max-range history. Any future case for native code has to start
from a fresh profile, not from the language.
**Scope:** the FastAPI backend in `src/`. The web app (TypeScript) and the macOS/iOS app (Swift)
use the backend only over HTTP through the OpenAPI types, so neither depends on the backend's
language.

---

## Verdict

**Neither language is worth moving to. If a full rewrite were ever forced, Rust fits this
codebase better than Go.**

- **Go** is strongest at the HTTP layer, which is the easy part of Investa. It has no mature
  dataframe library, so the time-series alignment that pandas does today would all be
  hand-written: reindexing, forward-fill, and timezone conversion.
- **Rust** has Polars, a dataframe library as capable as pandas. Its type system also suits
  currency and calendar-day correctness. But it is slower to change, and neither language has an
  equivalent of yfinance, pandas-market-calendars or pdfplumber.

Bringing Rust in for the hot calculations only, via PyO3 (Rust's Python bindings), was also
considered and declined. It would replace the code that is already the fastest part of the
request.

---

## Part 1: What a move would cover (measured 11 Oct 2026)

| | Size |
|---|---|
| `src/` Python | ~59,400 lines; 80 routes |
| `tests/` | ~21,200 lines, 86 files (pytest; they call Python functions directly, so none would port) |
| pandas | 37 files, ~820 call sites |
| NumPy | ~710 call sites |
| Date and timezone handling | ~160 `tz_localize`/`tz_convert`, ~95 `to_datetime` |
| Numba | 3 kernels in [`portfolio_valuation_kernels.py`](../src/portfolio_valuation_kernels.py#L34), 1 in [`portfolio_analyzer.py`](../src/portfolio_analyzer.py#L141) |

These dependencies have **no maintained equivalent in Go or Rust**:

- **yfinance:** an unofficial Yahoo scraper. A rewrite would mean owning a scraper against an
  undocumented, changing endpoint.
- **pandas-market-calendars:** exchange holiday calendars.
- **pdfplumber:** table extraction for broker statements in `server/pdf_parser.py`.

Gemini has an official Go SDK but not a Rust one. statsmodels (one OLS regression) and
`scipy.optimize` would port easily.

## Part 2: Where the time actually goes

Profiled the path `GET /api/history` calls: a cold max-range history for `kitmatan`
(5,024 transactions, THB, benchmarks off, `force=True` to bypass the result caches).

| Before the fix | Seconds |
|---|---|
| Whole computation | 18.6 |
| Market-data sync ([`get_historical_data`](../src/market_data.py#L4938)) | 15.7 |
| `check_integrity`: 44 queries with `date IN (~11,000 params)` | 6.4 |
| yfinance worker (network) | 4.8 |
| `DatetimeIndex.strftime` (four calls per symbol) | 2.8 |
| `upsert_ohlcv` | 2.5 |
| FX history | 1.9 |
| **Numba kernels** | **< 0.1** |

The one real cost of Numba is **compile time**. With an empty cache, the characterization tests
take 13.8s against 5.7s warm, so compilation adds about 8s. Numba's cache key includes the
source line, so any edit to the kernels file shifts the line numbers and throws the cache away.
On the day of the evaluation there were 63 stale `.nbi`/`.nbc` files in `src/__pycache__/`.
This costs one slow request per edit or fresh deploy.

## Part 3: What Rust would and would not buy

**Would buy:**
- **No first-call compile delay.** Rust is compiled ahead of time.
- **Freer upgrades.** Python and NumPy upgrades would no longer wait on Numba (currently Numba
  0.61.2, NumPy 2.0.0, Python 3.12).
- **Strict IEEE arithmetic.** The kernels run with `fastmath=True`, which lets the compiler
  reorder floating-point sums.

**Would not buy:** any user-visible speed, because the kernels are already under 0.1s.

**Would cost:**
- A second toolchain (cargo and maturin) on every machine that builds, plus CI changes and a
  Dockerfile stage.
- Builds for both macOS arm64 and Linux.
- Debugging across the Python/Rust boundary.
- Slower changes to ~1,800 lines of kernel code.
- Slower research. The ranking, intrinsic-value and backtest work is exploratory, and pandas is
  the right tool for it.

## Part 4: What was done instead

Two changes in [`market_db.py`](../src/market_db.py):

1. **[`check_integrity`](../src/market_db.py#L1321) reads a date range.** It uses
   `date BETWEEN ? AND ?` in place of `date IN (...)`. The `IN` form made SQLite probe the
   `(date, symbol)` index once per date. The range form is one contiguous scan of the primary
   key, measured 4–9× faster on the 3.2 GB archive. The existing inner merge on the incoming days
   keeps the comparison to exactly the same rows.
2. **[`_iso_days`](../src/market_db.py#L52) replaces `strftime("%Y-%m-%d")`** on the sync and
   read-adjustment paths. It truncates to whole days in NumPy, about 15× faster. Its output is
   identical for timezone-aware indexes (it uses the local wall date), pre-1970 instants and NaT.

**Equivalence check:** the old (`HEAD`) and new `check_integrity` were run side by side on 1,383
inputs across 154 real symbols. The inputs included unsorted, timezone-shifted, sparse, gapped,
future-dated and deliberately corrupted frames. There were **0 mismatches**, including the reason
text on all 227 inconsistent cases. The full suite (1,085 tests) passed. Two regression tests
were added to `tests/test_market_archive.py`.

| After the fix | Before | After |
|---|---|---|
| `check_integrity` | 6.39s | 0.47–1.20s |
| `strftime` | 2.77s | ~0 |
| `upsert_ohlcv` | 2.51s | 1.31s |
| `upsert_actions` | 0.77s | 0.13–0.17s |
| **Whole computation** | **18.6s** | **10.6s** |

About 5s of what remains is the yfinance network fetch, which varies from run to run.

## Part 5: When to revisit

Reopen this decision only if one of these becomes true, and **profile first**:

1. **The Buffett batch pipeline turns out to be CPU-bound.** `buffett_pipeline.py` and EDGAR XBRL
   parsing were not profiled. It is the one heavy, self-contained batch job that does not depend
   on yfinance, and the best candidate for a separate Rust job.
2. **The kernels become a measured bottleneck.** For example, many concurrent users, or Monte
   Carlo projections on every request.
3. **Memory caps how far the server scales**, and switching the heavy pandas paths to Polars
   (still in Python) does not fix it. Today the user-data loader and the fetch worker are already
   guarded against running out of memory.
4. **Numba blocks an upgrade that is needed**, such as a Python release it does not yet support.

Remaining cheap wins, all in Python: `upsert_ohlcv` (1.3s) and FX history (1.9s).

## Reproducing the profile

```bash
cd src && python ../scripts/profile_history.py kitmatan max
```

The script overwrites `profiles/history_<user>_<period>.prof`. To keep that file, copy the
script and point its output elsewhere. To time a cold Numba compile without touching the real
cache, set `NUMBA_CACHE_DIR` to an empty directory.
