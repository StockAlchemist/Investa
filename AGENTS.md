# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What This Is

Investa is a financial portfolio management system: a Next.js 16 / React 19 PWA (`web_app/`) on a single Python FastAPI backend (`src/`). A native SwiftUI macOS client (`macos_app/`) is in early development — it's a thin client of the same FastAPI backend (see `macos_app/README.md`).

> **Feature parity (required):** Investa has three clients — the **web app** (`web_app/`), the native **macOS app**, and the native **iOS/iPhone app** (macOS and iOS share the SwiftUI codebase in `macos_app/`). Any new user-facing function or feature MUST be implemented in **all three** (web, macOS, and iOS), not just one. Keep `web_app/lib/api.ts` and the SwiftUI API client in sync with the backend when adding endpoints.

## Running the App

```bash
# Full stack (backend on :8000, frontend on :3000)
./start_investa.sh

# Backend only
cd src && uvicorn server.main:app --reload --port 8000

# Frontend only
cd web_app && npm run dev
```

## Commands

### Python backend
```bash
pip install -r requirements.txt          # server/runtime deps
pip install -r requirements-dev.txt      # + pytest, ruff

pytest tests/                              # all tests
pytest tests/test_finutils.py -v          # single file
pytest tests/ -k "test_name" -v           # single test

ruff check src/                            # lint
ruff format src/                           # format
```

### Frontend (`web_app/`)
```bash
npm install
npm run dev          # dev server
npm run build        # production build
npm run lint         # ESLint
npm test             # Vitest unit tests (tests/unit/)
npm run test:e2e     # Playwright E2E tests (hermetic — backend mocked via page.route)
```

## Architecture

### Backend (`src/`)

The backend separates concerns across domain routers and core engine modules. Key files:

| File | Role |
|------|------|
| `server/main.py` | FastAPI app setup, lifespan, CORS, GZip & timing middleware |
| `server/api.py` | Top-level router aggregating domain routers under `server/routes/*` |
| `server/routes/*` | Domain REST routers (`portfolio`, `analytics`, `transactions`, `market`, `screener`, `watchlist`, `buffett_rank`, `strategies`, `settings`, `auth`, `admin`) |
| `server/portfolio_service.py` | Portfolio summary/history calculation engine, SWR caches, and background warmup |
| `portfolio_logic.py` | Core portfolio engine — positions, P&L, cash flows |
| `portfolio_history.py` | Historical performance and daily valuation series |
| `portfolio_valuation_kernels.py` | Numba JIT-accelerated portfolio valuation kernels |
| `portfolio_analyzer.py` | Performance metrics, returns, FIFO lots, drawdown |
| `market_data.py` | yfinance/Tiingo/FMP integration, bounded TTL caches, isolated worker pool |
| `market_db.py` | Persistent SQLite archive for daily/intraday OHLCV, FX, and corporate actions |
| `db_utils.py` | SQLite connection factory (`WAL` + `busy_timeout`), schema migrations, user/screener CRUD |
| `finutils.py` | Low-level financial math |
| `financial_ratios.py` | Twelve intrinsic-value models, the business-aware blend, financial ratios |
| `server/refresh_worker.py` | Background tasks (price refresh, index quotes, Thai fund NAVs) |
| `server/ai_analyzer.py` | Per-stock Gemini AI analysis |
| `server/portfolio_ai_analyzer.py` | Whole-portfolio Gemini AI review and optimization analysis |
| `server/screener_service.py` | Market screener logic |
| `buffett_pipeline.py` | Buffett/value ranking of all US listed stocks (batch) |
| `edgar_provider.py` | SEC EDGAR XBRL fundamentals (~17y annual + quarterly history, tag fallback chains) |
| `ibkr_connector.py` | Interactive Brokers Flex Web Service sync |

Performance-critical valuation paths use **Numba JIT** (`@jit(nopython=True)`). Avoid breaking Numba-compatible code (no Python objects, use NumPy arrays).

### Frontend (`web_app/`)

Next.js App Router layout:
- `app/page.tsx` — Dashboard (main entry)
- `app/(auth)/` — Login/register pages
- `app/screener/` — Market screener
- `components/` — Modular React components and feature subdirectories (`dashboard/`, `holdings/`, `stock-detail/`, `transactions/`, `performance/`, `portfolio/`, `capital-gains/`, `income/`, `watchlist/`, `settings/`, `layout/`, `ui/`)
- `lib/api.ts` — Typed API client backed by OpenAPI types (`src/api/types.ts`)
- `context/` — Auth and theme React contexts

Data fetching uses **TanStack Query** (React Query). UI primitives come from **Radix UI** with **Tailwind CSS** styling. Charts use **Recharts**.

### Data & Storage

- `data/db/` — SQLite databases (user data, screener cache)
- `data/cache/` — Market data cache (yfinance responses)
- `data/config/` — `gui_config.json`, `manual_overrides.json`
- `data/users/` — Per-user storage

## Environment

Copy `.env` and populate:
```
GEMINI_API_KEY=...
FMP_API_KEY=...       # Financial Modeling Prep (optional)
IBKR_PORT=...         # Interactive Brokers (optional)
AUTH_SECRET_KEY=...   # JWT signing key (optional; auto-generated to data/config/auth_secret.key if unset)
CORS_ALLOW_ORIGINS=... # Extra CORS origins, comma-separated (local/LAN/Tailscale covered by default)
INVESTA_LOG_LEVEL=...  # Application log level (default WARNING; set INFO/DEBUG when debugging)
```

## Key Conventions

- **Python**: Ruff for linting/formatting; type hints used throughout via Pydantic models.
- **TypeScript**: Strict mode; `lib/api.ts` mirrors backend Pydantic models — keep them in sync when changing API contracts.
- **Database migrations**: Run in isolation via scripts in `scripts/`; never modify the schema directly against a live DB.
- **Numba**: JIT-compiled functions in `financial_ratios.py` and `portfolio_logic.py` must use NumPy-compatible code only.
- **Multi-user**: The backend supports per-user data isolation; user context flows through FastAPI dependencies in `server/dependencies.py`.
- **UI**: The primary UI is the web app (`web_app/components/*.tsx`); e.g. the transaction form is `web_app/components/TransactionModal.tsx`. There is no longer a desktop/Qt UI. A native SwiftUI macOS client lives in `macos_app/` (SwiftUI + MVVM, async URLSession, generated via XcodeGen from `macos_app/project.yml`); it consumes the same FastAPI endpoints as the web app and mirrors `web_app/lib/api.ts`.
- **Git**: Use HTTPS remotes, or fallback to HTTPS if SSH connection/authentication is unavailable or restricted.
