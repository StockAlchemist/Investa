# Investa — Native macOS App

A native SwiftUI macOS client for the Investa FastAPI backend, with a sidebar
shell hosting these tabs:

- **Dashboard** — login, control bar (account/currency/period), metric cards,
  holdings table, and a Swift Charts performance graph.
- **Transactions** — full CRUD (add/edit/delete) with the backend's signed
  Total Amount logic replicated from the web modal.
- **Dividends** — read-only table with display-currency totals.
- **Capital Gains** — read-only realized-gains table.
- **Watchlist** — view items plus add/remove (default watchlist).

It exercises the full stack — auth → networking → models → SwiftUI + Swift
Charts — and is the skeleton the remaining tabs (screener, stock detail, AI
analysis, settings editing, …) are added onto.

The app is a **client**; it does not bundle the Python backend. Run the backend
separately and point the app at it (default `http://localhost:8000/api`, editable
from the login screen's "Server" button).

## Build & run

The Xcode project is generated from `project.yml` with
[XcodeGen](https://github.com/yonyz/XcodeGen) so it stays reviewable in git.

```bash
# 1. One-time: install the project generator
brew install xcodegen

# 2. Generate Investa.xcodeproj (re-run whenever project.yml or the file tree changes)
cd macos_app
xcodegen generate

# 3. Start the backend (from the repo root, in another terminal)
./start_investa.sh
# or: cd src && uvicorn server.main:app --reload --port 8000

# 4a. Open in Xcode and Run (⌘R)
open Investa.xcodeproj

# 4b. …or build from the CLI
xcodebuild -project Investa.xcodeproj -scheme Investa -destination 'platform=macOS' build
```

Generated artifacts (`Investa.xcodeproj/`, build output) are not meant to be
committed — add them to `.gitignore` if desired.

## Architecture

- **Networking/** — `APIClient` (async URLSession, bearer auth, 401 → `.authExpired`),
  `KeychainStore` (token persistence), `JSONValue` (decodes the backend's
  dynamic, currency-suffixed keys), `APIConfig` (base URL).
- **Models/** — `User`, `Metrics`, `Holding`, `PerformancePoint`, `AppSettings`.
  Metrics/holdings keep the raw `[String: JSONValue]` and expose typed accessors,
  resolving currency-suffixed columns (e.g. `"Market Value (USD)"`) at runtime.
- **Auth/** — `AuthViewModel` (login via OAuth2 form post, session restore,
  logout) + `LoginView`.
- **App/** — `InvestaApp` (`@main`, menu-bar commands), `AppState` (shared
  currency/account/period selection), `RootView` (login ↔ dashboard router),
  `Formatters`.
- **Features/Dashboard/** + **Features/ControlBar/** — the dashboard UI.

## What's next

Screener, stock detail modal, AI analysis/chat, settings editing, IBKR sync,
document parsing. Each is a new `Features/<Tab>/` module reusing `APIClient` and
the `JSONValue` decoding approach. Bundling the Python backend into the `.app`
is a separate, larger effort.
