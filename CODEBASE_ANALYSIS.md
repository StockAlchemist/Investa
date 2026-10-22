# Investa Codebase Review & Analysis Report

**Date:** October 2026
**Scope:** Full Stack (Python FastAPI Backend, Next.js 16 / React 19 Frontend, Native macOS & iOS SwiftUI Client)  
**Status:** Updated through October 2026 architecture and hardening pass.

---

## 1. Executive Summary

Investa is a mature, well-architected financial portfolio management platform comprising:
- **Backend (`src/`):** Python FastAPI application with domain routers (`src/server/routes/*`), shared SWR calculation engine (`src/server/portfolio_service.py`), Numba-accelerated valuation kernels, SQLite storage with multi-user isolation, yfinance / Tiingo / FMP / EDGAR integration, and automated background workers.
- **Web App (`web_app/`):** Next.js 16 App Router PWA built with React 19, TypeScript, TanStack Query, Radix UI, Tailwind CSS, and Recharts, organized into modular feature subdirectories under `web_app/components/`.
- **Native macOS / iOS App (`macos_app/`):** Universal SwiftUI client with MVVM architecture, async/await networking with exponential backoff and durable transaction idempotency keys (`APIClient.swift`), and native App Intents.

---

## 2. Completed Architectural & Hardening Milestones

### A. Backend (`src/`)
1. **Domain Route Decomposition:** Split the legacy `src/server/api.py` monolith into focused routers under `src/server/routes/` (`portfolio`, `analytics`, `transactions`, `market`, `screener`, `watchlist`, `buffett_rank`, `strategies`, `settings`, `auth`, `admin`) backed by `src/server/portfolio_service.py`.
2. **SQLite Connection Factory & Pragmas:** Standardized database connections via `db_utils.get_db_connection()` enforcing `PRAGMA busy_timeout = 30000`, `PRAGMA foreign_keys = ON`, and `WAL` mode across `BuffettRankStore`, `EdgarFactStore`, and `MarketDatabase`.
3. **Bounded In-Memory Caches & SWR:** Replaced unbounded quote/statement dictionaries in `src/market_data.py` with `BoundedTTLCache` and added `SWRCache` in `src/server/route_utils.py` for non-blocking stale-while-revalidate summary and history serving.
4. **Market-Data Sync & FX Vectorization:** Replaced `date IN (...)` integrity probes with contiguous `date BETWEEN ? AND ?` scans and vectorized date/rate extraction (`_iso_days`) in `src/market_db.py` across both `upsert_ohlcv` and `upsert_fx`.
5. **Market-Local Clock Enforcement:** Standardized fallback trade/corporate-action dates in `src/ibkr_connector.py` and AI review cache keys/prompts in `src/server/ai_analyzer.py` and `src/server/portfolio_ai_analyzer.py` on `get_est_today()` / `get_est_now()`.
6. **Route Optimization & Error Sanitization:** Removed redundant full-history calculation in `POST /portfolio/ai_review`, switched `GET /projected_income` to share `_compute_raw_summary`, added `ExternalID` duplicate protection to `POST /sync/ibkr/approve`, and sanitized 500/error payloads across `src/server/portfolio_service.py` and `src/server/routes/analytics.py`.

### B. Web Frontend (`web_app/`) & Native Client (`macos_app/`)
1. **Modular Component Hierarchy:** Decomposed `StockDetailModal`, `HoldingsTable`, `Settings`, and `TransactionsTable` into dedicated feature sub-packages (`components/stock-detail/`, `components/holdings/`, `components/settings/`, `components/transactions/`).
2. **OpenAPI Type Drift Guard:** Added offline OpenAPI schema export (`scripts/export_openapi.py`) and `npm run check:api-drift` to keep `web_app/src/api/types.ts` aligned with FastAPI response models.
3. **Native Networking Resilience:** Added exponential backoff retries, durable `Idempotency-Key` submission tracking, and toast notifications in `macos_app/Investa/Networking/APIClient.swift`.

### C. Build & Container Hygiene
1. **Multi-Stage Docker Build & Data Isolation:** Updated `Dockerfile` to a multi-stage build (omitting `gcc`/`g++` from the runtime image) and excluded `data` in `.dockerignore` so user portfolios, `global.db`, and `auth_secret.key` are never baked into container layers.
2. **Configurable Dev Origins & Clean Dependencies:** Parameterized `allowedDevOrigins` via `NEXT_ALLOWED_DEV_ORIGINS` in `web_app/next.config.ts`.


## 3. Release validation — 22 October 2026

The combined release candidate passed:

- Backend: 1,128 tests; Ruff on changed Python files; mypy on all 17 configured modules.
- Web: 249 unit tests, 75 Chromium end-to-end tests, and 11 iPhone/WebKit checks.
- Frontend production build and ESLint on changed files.
- Generated OpenAPI type comparison with the checked-in API types.
- Native: 115 macOS tests and 97 iOS simulator tests.
- The changed private portfolio database passed SQLite integrity checks; its commit uses the private repository's WAL checkpoint hook.

Validation also corrected foreground timeout handling during background warmup,
completed live-chart input wiring, added the web layout reset control for client
parity, and covered FIFO cache isolation and same-day intraday round trips.

The Dockerfile and data exclusions were reviewed. A container image build was
not run because Docker is unavailable on the validation host.
