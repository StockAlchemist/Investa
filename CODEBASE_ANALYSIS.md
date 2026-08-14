# Investa Codebase Review & Analysis Report

**Date:** August 2026  
**Scope:** Full Stack (Python FastAPI Backend, Next.js 16 / React 19 Frontend, Native macOS & iOS SwiftUI Client)  
**Status:** Review Only — No code changes made.

---

## 1. Executive Summary

Investa is a mature, well-architected financial portfolio management platform comprising:
- **Backend (`src/`):** Python FastAPI application with Numba-accelerated valuation kernels, SQLite database storage with multi-user isolation, yfinance / EDGAR integration, and automated background workers.
- **Web App (`web_app/`):** Next.js 16 App Router PWA built with React 19, TypeScript, TanStack Query, Radix UI, Tailwind CSS, and Recharts.
- **Native macOS / iOS App (`macos_app/`):** Universal SwiftUI client with MVVM architecture, async/await networking, and native App Intents.

Both automated test suites passed completely:
- **Python Backend:** **729 passed** (`pytest tests/`)
- **Web Frontend:** **152 passed** (`vitest run`), **0 lint errors** (`eslint`)
- **Swift macOS Target:** **Build Succeeded** (`xcodebuild Investa`)
- **Swift iOS Target:** **Build Succeeded** (`xcodebuild Investa-iOS`)

This analysis highlights identified bugs, potential edge cases, structural inconsistencies, and recommended architectural enhancements across all layers.

---

## 2. Detailed Findings

### A. Backend (`src/`)

#### 1. Concurrency & SQLite Connection Contention (Medium Risk)
- **Issue:** Background jobs (`market_data_worker.py`, `screener_worker.py`, `buffett_pipeline.py`) and incoming FastAPI request threads concurrently open SQLite connections to `market_data.db`, `screener_cache.db`, and user databases.
- **Risk:** While WAL mode is enabled during startup in `db_utils.py`, some direct connection initializations across worker modules do not consistently set `PRAGMA busy_timeout = 30000;`. Under heavy batch processing, transient `sqlite3.OperationalError: database is locked` errors can arise.
- **Recommendation:** Standardize all DB connections through a single connection factory in `db_utils.py` that guarantees `busy_timeout` and WAL pragma configuration.

#### 2. Exception Handling & Error Detail Exposure (Low-Medium Risk)
- **Issue:** In several route modules (`src/server/routes/transactions.py`, `src/server/routes/screener.py`, `src/server/routes/portfolio.py`), exceptions are caught and returned as:
  ```python
  raise HTTPException(status_code=500, detail=str(e))
  ```
- **Risk:** This can expose internal filesystem paths, SQL syntax errors, or database schema names to API clients in production.
- **Recommendation:** Return sanitized, user-friendly error messages in API responses while logging full tracebacks to application logs.

#### 3. In-Memory Cache Growth & Monotonic Memory Footprint (Medium Risk)
- **Issue:** `market_data.py` maintains in-memory price dictionaries (`_GLOBAL_PRICE_CACHE`, historical quote caches, intraday buffers) and ticker mappings.
- **Risk:** For long-running server instances monitoring large universes (e.g. S&P 500 + Russell 2000 + Watchlists), in-memory cache size grows monotonically without an explicit size cap or TTL eviction cycle.
- **Recommendation:** Implement bounded LRU caches with time-to-live expiration (similar to `SWRCache` used in `server/route_utils.py`) for in-memory market structures.

#### 4. Edge Cases in Zero-Division / Micro-Quantity (Dust) Portfolios (Low Risk)
- **Issue:** When positions have near-zero quantity remaining (e.g. `1e-7` shares due to floating point fractions or dust after full liquidation), calculations of average cost, percentage return, or annualized CAGR in `portfolio_logic.py` and `portfolio_valuation_kernels.py` rely on floating comparison thresholds (e.g., `abs(qty) < 1e-9`).
- **Risk:** Micro-holdings or positions held for less than 1 calendar day can produce extreme annualized returns or `NaN`/`Inf` if not strictly clamped before serialization.
- **Recommendation:** Ensure all ratio / performance outputs pass through `clean_nans` and enforce minimum duration bounds (e.g., 1 day) for annualized return compounding.

---

### B. Web Frontend (`web_app/`)

#### 1. Large Monolithic Component Files & Bundle Splitting (Improvement)
- **Observation:** Key components are very large:
  - `StockDetailModal.tsx` (~192 KB / ~3,000 LOC)
  - `HoldingsTable.tsx` (~117 KB / ~2,200 LOC)
  - `Settings.tsx` (~105 KB / ~2,000 LOC)
  - `TransactionsTable.tsx` (~92 KB / ~1,800 LOC)
- **Impact:** While functioning properly, maintaining and testing these monolithic files is difficult. Initial bundle parsing and React re-renders on table mutations can be optimized.
- **Recommendation:** Decompose into smaller focused sub-components (e.g., separate tabs, sub-modals, column renderers, action dialogs) using dynamic imports / `React.lazy` for heavy modal tabs.

#### 2. Query Invalidation Precision (Improvement)
- **Observation:** In several mutation callbacks (e.g., creating/updating a transaction, updating manual valuation), queries are invalidated broadly (`queryClient.invalidateQueries({ queryKey: ['portfolio'] })`).
- **Impact:** Triggers multiple simultaneous background network requests even for unaffected sub-views.
- **Recommendation:** Refine query invalidation keys to target specific active views and leverage optimistic updates where appropriate.

#### 3. API Model Alignment (`web_app/lib/api.ts`) (Inconsistency)
- **Observation:** `web_app/lib/api.ts` contains manual TypeScript interfaces alongside `apiClient` generated types. Some optional fields (e.g. `fx_gain_loss_display`, `annualized_twr`, `taxes`, `benchmark_scoreboard`) have nullable/optional typing differences compared to backend Pydantic models.
- **Recommendation:** Standardize typing using OpenAPI-generated client schemas or maintain single-source typing across models.

---

### C. Native macOS & iOS Client (`macos_app/`)

#### 1. Feature Parity with Web App (High Quality, Minor Gaps)
- **Status:** The native SwiftUI client in `macos_app/` achieves strong parity across Dashboard, Holdings, Transactions, Markets, SP500 Heatmap, Screener, Buffett Rank, Strategies, and Settings.
- **Identified Gaps / Inconsistencies:**
  - **Account Group Management:** The web client allows creating and custom grouping accounts with specific sorting/filtering; in SwiftUI, account filtering is available, but custom group creation is managed primarily on web.
  - **Layout Customization:** Web client provides modular widget drag-and-drop / ordering via `LayoutConfigurator.tsx`; SwiftUI uses a fixed layout grid.
  - **Error Display:** SwiftUI handles errors via inline view states, while some network failures (e.g., background price refresh timeout) fail silently without a global toast notification.

#### 2. Network Client Resilience (`APIClient.swift`)
- **Observation:** `APIClient.swift` has a hardcoded 300s timeout for slow endpoints and 60s for standard endpoints.
- **Improvement:** Implement automatic exponential backoff retry for transient transport errors (e.g. backend restarting or network transitions between Wi-Fi and cellular on iOS).

---

## 3. Prioritized Recommendations Roadmap

| Priority | Category | Task | Impact |
| :--- | :--- | :--- | :--- |
| **P1** | **Backend / DB** | Standardize SQLite connection timeouts and WAL pragmas across all background workers | Prevents rare DB lock contention during concurrent syncs |
| **P1** | **Backend / Security** | Sanitize 500 error messages returned by route handlers | Enhances API security and prevents internal detail leaks |
| **P2** | **Backend / Memory** | Implement TTL / LRU bounds on in-memory price dictionaries in `market_data.py` | Prevents long-term memory growth |
| **P2** | **Frontend / Web** | Refactor monolithic components (`StockDetailModal`, `HoldingsTable`, `Settings`) into smaller modular subcomponents | Improves maintainability, bundle size, and re-render performance |
| **P3** | **Native / Swift** | Add toast error banner and retry logic in `APIClient.swift` | Enhances iOS and macOS offline / reconnection resilience |
| **P3** | **Parity / UX** | Expand account grouping management in native SwiftUI client | Reaches 100% parity with web client configuration capabilities |

---

## 4. Conclusion

The Investa codebase is in excellent health:
- Core financial algorithms and performance calculations are solid and well-covered by tests.
- Multi-user data isolation is implemented cleanly via per-user SQLite databases.
- Multi-platform support (Web, macOS, iOS) is functional with high feature parity.
- The identified items represent hardening, architectural polish, and performance optimizations.
