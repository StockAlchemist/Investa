# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Investa is a financial portfolio management system with three deployment targets sharing a single Python FastAPI backend:

- **Web app** — Next.js 16 / React 19 PWA (`web_app/`)
- **Desktop app** — Electron wrapper around the built web app (`desktop-electron/`)
- **Legacy GUI** — PySide6 (Qt) desktop app (`src/main_gui.py`), largely in maintenance mode

## Running the App

```bash
# Full stack (backend on :8000, frontend on :3000)
./start_investa.sh

# Backend only
cd src && uvicorn server.main:app --reload --port 8000

# Frontend only
cd web_app && npm run dev

# Legacy Qt GUI
python src/main_gui.py

# Electron desktop (requires built frontend)
cd web_app && npm run build:desktop
./start_desktop.sh
```

## Commands

### Python backend
```bash
pip install -r requirements.txt

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
npm run build:desktop  # static export for Electron (outputs to out/)
npm run lint         # ESLint
npm run test:e2e     # Playwright E2E tests
```

### Electron (`desktop-electron/`)
```bash
npm start            # dev run
npm run pack         # test packaging
npm run dist         # full distributable build
```

## Architecture

### Backend (`src/`)

The backend separates concerns across large modules. Key files:

| File | Role |
|------|------|
| `server/main.py` | FastAPI app setup, lifespan |
| `server/api.py` | All REST routes (~3,600 LOC) |
| `portfolio_logic.py` | Core portfolio engine — positions, P&L, cash flows (~7,300 LOC) |
| `portfolio_analyzer.py` | Performance metrics, returns, drawdown |
| `market_data.py` | yfinance integration, price caching, fallbacks |
| `db_utils.py` | SQLite ORM, all database operations |
| `models.py` | Pydantic request/response models |
| `finutils.py` | Low-level financial math |
| `financial_ratios.py` | DCF valuation, financial ratios |
| `workers.py` | Background tasks (price refresh, etc.) |
| `server/ai_analyzer.py` | Per-stock Gemini AI analysis |
| `server/screener_service.py` | Market screener logic |
| `ibkr_connector.py` | Interactive Brokers sync |

Performance-critical valuation paths use **Numba JIT** (`@jit(nopython=True)`). Avoid breaking Numba-compatible code (no Python objects, use NumPy arrays).

### Frontend (`web_app/`)

Next.js App Router layout:
- `app/page.tsx` — Dashboard (main entry)
- `app/(auth)/` — Login/register pages
- `app/screener/` — Market screener
- `components/` — 51 React components; key ones: `Dashboard.tsx`, `HoldingsTable.tsx`, `PerformanceGraph.tsx`, `StockDetailModal.tsx`
- `lib/api.ts` — Typed API client
- `context/` — Auth and theme React contexts

Data fetching uses **TanStack Query** (React Query). UI primitives come from **Radix UI** with **Tailwind CSS** styling. Charts use **Recharts**.

### Data & Storage

- `data/db/` — SQLite databases (user data, screener cache)
- `data/cache/` — Market data cache (yfinance responses)
- `data/config/` — `gui_config.json`, `manual_overrides.json`
- `data/users/` — Per-user storage

The desktop Electron app spawns its own Python backend on port **8001** (not 8000) to avoid conflicts with a running web instance.

## Environment

Copy `.env` and populate:
```
GEMINI_API_KEY=...
FMP_API_KEY=...       # Financial Modeling Prep (optional)
IBKR_PORT=...         # Interactive Brokers (optional)
```

## Key Conventions

- **Python**: Ruff for linting/formatting; type hints used throughout via Pydantic models.
- **TypeScript**: Strict mode; `lib/api.ts` mirrors backend Pydantic models — keep them in sync when changing API contracts.
- **Database migrations**: Run in isolation via scripts in `scripts/`; never modify the schema directly against a live DB.
- **Numba**: JIT-compiled functions in `financial_ratios.py` and `portfolio_logic.py` must use NumPy-compatible code only.
- **Multi-user**: The backend supports per-user data isolation; user context flows through FastAPI dependencies in `server/dependencies.py`.
