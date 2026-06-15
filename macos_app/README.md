# Investa — Native macOS App

A native SwiftUI macOS client for the Investa FastAPI backend, aiming for feature
parity with the web app. The sidebar mirrors the web app's tabs (same order), and
each tab composes the same widgets the web shows:

- **Performance** — control bar, 18 metric cards, performance graph
  (Value/TWR/Drawdown toggle + benchmark overlay + period), portfolio health,
  risk metrics, sector & top-contributor attribution, allocation donuts, dividend
  calendar, and the full holdings table.
- **Watchlist** — items + add/remove (default watchlist), with detail on click.
- **Market** — screener: run by universe (watchlist/manual/S&P/etc.) or AI narrative.
- **Transactions** — full CRUD with the backend's signed Total Amount logic.
- **Allocation** — holdings table + donut breakdowns by sector / asset type /
  geography / industry / account / currency.
- **Asset Change** — period-over-period asset value series + detail grid.
- **Capital Gains** — unrealized gains by holding + realized-gains table.
- **Dividend** — projected income chart + dividend calendar + dividend history.
- **AI Review** — portfolio AI review (scorecard + recommendations) and chat.
- **Markets** (native extra) — index quotes + market news.
- **Settings** — currency, closed-account toggle, benchmarks, manual price
  overrides, backend URL, cache clear, change password, log out.

The holdings table carries ~22 columns + a 7d sparkline. **Stock Detail** opens
as a sheet from any holding/watchlist/screener row.

**Stock Detail** opens as a sheet (double-click / "View Details" on a holding,
watchlist, or screener row): fundamentals, price chart, intrinsic value, earnings,
and an on-demand AI review. **Register** is available from the login screen.

Auth → networking → models → SwiftUI + Swift Charts throughout.

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

Remaining web features not yet ported: IBKR sync (pending/approve/reject),
document/statement parsing upload, portfolio optimization & allocation drift,
account-group management and per-account closure-date/interest-rate editing,
target-allocation editing, and the command palette. The performance graph also
doesn't yet overlay benchmarks or toggle TWR/drawdown. Each is a new or expanded
`Features/<Tab>/` module reusing `APIClient` and the `JSONValue` decoding
approach. Bundling the Python backend into the `.app` is a separate, larger
effort.
