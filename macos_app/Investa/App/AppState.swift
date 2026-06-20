import Foundation
import Combine

/// History period options matching the backend's accepted `period` values
/// (mirrors the web `PeriodSelector`).
enum Period: String, CaseIterable, Identifiable, Sendable {
    case oneDay = "1d"
    case fiveDays = "5d"
    case oneMonth = "1m"
    case threeMonths = "3m"
    case sixMonths = "6m"
    case ytd = "ytd"
    case oneYear = "1y"
    case threeYears = "3y"
    case fiveYears = "5y"
    case tenYears = "10y"
    case all = "all"
    case custom = "custom"

    var id: String { rawValue }
    var label: String {
        switch self {
        case .oneDay: return "1D"
        case .fiveDays: return "5D"
        case .oneMonth: return "1M"
        case .threeMonths: return "3M"
        case .sixMonths: return "6M"
        case .ytd: return "YTD"
        case .oneYear: return "1Y"
        case .threeYears: return "3Y"
        case .fiveYears: return "5Y"
        case .tenYears: return "10Y"
        case .all: return "All"
        case .custom: return "Custom"
        }
    }

    /// Sampling interval for the history fetch (mirrors the web `graphInterval`).
    var interval: String {
        switch self {
        case .oneDay: return "2m"
        case .fiveDays: return "15m"
        default: return "1d"
        }
    }
}

/// Shared selection state driving every dashboard panel: display currency,
/// selected accounts, history period, and whether to include closed accounts.
/// Mirrors the web app's ControlBar/CurrencySelector/AccountSelector state.
@MainActor
final class AppState: ObservableObject {
    @Published var displayCurrency: String = "USD"
    @Published var availableCurrencies: [String] = ["USD"]
    @Published var allAccounts: [String] = []
    @Published var accountGroups: [String: [String]] = [:]
    @Published var accountGroupOrder: [String] = []
    /// Empty set means "all accounts" (no `accounts` query param sent).
    @Published var selectedAccounts: Set<String> = []
    @Published var period: Period = .oneYear
    /// Custom date range for the performance graph (used when `period == .custom`).
    @Published var customFrom: Date = Calendar.current.date(byAdding: .year, value: -1, to: Date()) ?? Date()
    @Published var customTo: Date = Date()
    @Published var showClosed: Bool = false
    @Published var benchmarks: [String] = []
    /// Target allocation per bucket (quoteType/sector/country/symbol) → name → %.
    /// Drives the Dashboard "Actionable Insights" drift detection.
    @Published var targetAllocation: [String: [String: Double]] = [:]
    @Published private(set) var didLoadSettings = false
    /// Per-tab Layout configurator state: web tab id → set of visible section ids.
    @Published var tabVisible: [String: Set<String>] = [:]
    /// Accounts marked closed (for the account-selector indicator).
    @Published var closedAccounts: Set<String> = []
    /// account → ISO currency code (e.g. "IBKR USD" → "USD").
    @Published var accountCurrencyMap: [String: String] = [:]
    /// account → cash mode: "Auto" or "Manual".
    @Published var accountCashModeMap: [String: String] = [:]
    
    /// Current exchange rate of displayCurrency to USD (if not USD).
    @Published var currentFXRateToUSD: Double? = nil

    /// Global market indices for the app title bar strip.
    @Published var indices: [IndexQuote] = []
    /// Whether the US market is currently open (nil until first fetch).
    @Published var marketIsOpen: Bool? = nil

    private let api: APIClient
    private let visibleDefaultsKey = "investa.tabVisible"

    init(api: APIClient = .shared) {
        self.api = api
        loadVisiblePersisted()
    }

    // MARK: - Layout configurator (per-tab visible sections)

    func isVisible(_ section: AppSection, _ id: String) -> Bool {
        let set = tabVisible[section.rawValue] ?? TabLayout.defaultVisible(for: section)
        return set.contains(id)
    }

    func toggle(_ section: AppSection, _ id: String) {
        var set = tabVisible[section.rawValue] ?? TabLayout.defaultVisible(for: section)
        if set.contains(id) {
            if set.count > 1 { set.remove(id) }   // keep at least one, like the web
        } else {
            set.insert(id)
        }
        tabVisible[section.rawValue] = set
        persistVisible()
    }

    /// Visible ids for a tab, in the registry's canonical order.
    func visibleOrdered(_ section: AppSection) -> [String] {
        let set = tabVisible[section.rawValue] ?? TabLayout.defaultVisible(for: section)
        return TabLayout.items(for: section).map(\.id).filter { set.contains($0) }
    }

    private func persistVisible() {
        let encodable = tabVisible.mapValues { Array($0) }
        if let data = try? JSONEncoder().encode(encodable) {
            UserDefaults.standard.set(data, forKey: visibleDefaultsKey)
        }
    }
    private func loadVisiblePersisted() {
        guard let data = UserDefaults.standard.data(forKey: visibleDefaultsKey),
              let decoded = try? JSONDecoder().decode([String: [String]].self, from: data) else { return }
        tabVisible = decoded.mapValues { Set($0) }
    }

    /// The `accounts` query value: nil when nothing (or everything) is selected.
    var accountsQuery: [String]? {
        selectedAccounts.isEmpty ? nil : Array(selectedAccounts)
    }

    private static let ymd: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    var customFromYMD: String { Self.ymd.string(from: customFrom) }
    var customToYMD: String { Self.ymd.string(from: customTo) }

    /// Update the benchmark selection and persist it to the backend settings.
    func setBenchmarks(_ list: [String]) {
        benchmarks = list
        struct Body: Encodable { let benchmarks: [String] }
        Task { let _: StatusResponse? = try? await api.send(method: "POST", path: "/settings/update", body: Body(benchmarks: list)) }
    }

    /// Load currency options, account list, and saved defaults from the backend.
    func loadSettings() async {
        do {
            let settings: AppSettings = try await api.get("/settings")
            availableCurrencies = settings.availableCurrencies?.isEmpty == false
                ? settings.availableCurrencies! : ["USD"]
            accountGroups = settings.accountGroups ?? [:]
            accountGroupOrder = settings.accountGroupOrder ?? Array(accountGroups.keys).sorted()
            if let cur = settings.displayCurrency, availableCurrencies.contains(cur) {
                displayCurrency = cur
            } else if !availableCurrencies.contains(displayCurrency) {
                displayCurrency = availableCurrencies.first ?? "USD"
            }
            showClosed = settings.showClosed ?? false
            benchmarks = settings.benchmarks ?? []
            targetAllocation = settings.targetAllocation ?? [:]
            // Accounts with a closure date on/before today are "closed".
            let today = ISO8601DateFormatter().string(from: Date()).prefix(10)
            closedAccounts = Set((settings.accountClosureDates ?? [:]).compactMap { acc, date in
                String(date.prefix(10)) <= today ? acc : nil
            })
            // Full account list comes from the summary's _available_accounts (all
            // transaction accounts, incl. closed) — union with closed + grouped.
            var accounts = Set<String>()
            if let headline: SummaryResponse = try? await api.get(
                "/summary/headline", query: [URLQueryItem(name: "currency", value: displayCurrency)]),
               let list = headline.metrics?.raw["_available_accounts"]?.arrayValue?.compactMap({ $0.stringValue }) {
                accounts.formUnion(list)
            }
            accounts.formUnion(settings.allAccounts)
            accounts.formUnion(closedAccounts)
            accounts.formUnion(accountGroups.values.flatMap { $0 })
            allAccounts = accounts.sorted()
            accountCurrencyMap = settings.accountCurrencyMap ?? [:]
            accountCashModeMap = settings.accountCashModeMap ?? [:]
            if let saved = settings.selectedAccounts, !saved.isEmpty {
                selectedAccounts = Set(saved.filter { allAccounts.contains($0) })
            }
            await fetchFXRate()
            await fetchIndices()
            didLoadSettings = true
        } catch {
            // Non-fatal: fall back to defaults so the dashboard still loads.
            didLoadSettings = true
            await fetchIndices()
        }
    }

    /// Fetch the exchange rate for the current display currency vs USD.
    func fetchFXRate() async {
        guard displayCurrency != "USD" else {
            currentFXRateToUSD = nil
            return
        }
        struct FXRateResponse: Codable { let rate: Double? }
        let currentCur = displayCurrency
        do {
            let res: FXRateResponse = try await api.get("/fx_rate/\(currentCur)")
            if self.displayCurrency == currentCur {
                self.currentFXRateToUSD = res.rate
            }
        } catch {
            print("Failed to fetch FX rate: \(error)")
        }
    }

    /// Fetch the global market indices for the app title bar.
    func fetchIndices() async {
        do {
            if let map: [String: IndexQuote] = try await api.get("/indices") {
                let preferredOrder = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000", "VIX"]
                indices = Array(map.values).sorted { (a: IndexQuote, b: IndexQuote) -> Bool in
                    let iA = preferredOrder.firstIndex(of: a.name ?? "") ?? 99
                    let iB = preferredOrder.firstIndex(of: b.name ?? "") ?? 99
                    if iA != iB { return iA < iB }
                    return (a.name ?? "") < (b.name ?? "")
                }
            }
        } catch {
            print("Failed to fetch indices: \(error)")
        }
        if let status: MarketStatusResponse = try? await api.get("/market_status") {
            marketIsOpen = status.is_open
        }
    }
}

/// `GET /market_status` → `{ "is_open": bool }`.
private struct MarketStatusResponse: Decodable, Sendable {
    let is_open: Bool
}
