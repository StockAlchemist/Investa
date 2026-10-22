import Foundation

/// Loads and holds the Performance tab's data: headline metrics, full summary,
/// holdings, performance history (with benchmarks), risk metrics, portfolio
/// health, attribution, and the dividend calendar. Refetches when the shared
/// selection (currency / accounts / period / benchmarks) changes.
@MainActor
final class DashboardViewModel: ObservableObject {
    @Published var metrics: Metrics?
    @Published var holdings: [Holding] = []
    @Published var history: [PerformancePoint] = []
    /// 1-year daily history used by the hero card's WTD/MTD/YTD/1Y selector,
    /// independent of the global period that drives the performance graph.
    @Published var heroHistory: [PerformancePoint] = []
    @Published var heroIntraday: [PerformancePoint] = []
    @Published var heroWTD: [PerformancePoint] = []
    @Published var risk: RiskMetrics?
    @Published var projection: Projection?
    /// Loaded only when the user opens the projection card's Backtest tab —
    /// the walk-forward refit is far too heavy to run on every dashboard load.
    @Published var projectionBacktest: ProjectionBacktest?
    @Published var isLoadingBacktest = false
    @Published var health: PortfolioHealth?
    @Published var attribution: Attribution?
    @Published var dividendEvents: [DividendEvent] = []
    @Published var earningsEvents: [EarningsEvent] = []

    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var widgetErrors: [String: String] = [:]
    @Published var calculatedAt: Date?
    @Published var backendRefreshing = false
    private var generation = UUID()
    private var loadedScope: String?
    private var chartCache: [String: [PerformancePoint]] = [:]

    private let api: APIClient
    private var loadTask: Task<Void, Never>?
    /// Selection the loaded backtest belongs to, so re-opening the tab doesn't refetch.
    private var backtestKey: String?
    /// Whether the user has opened the Backtest tab this session — once they
    /// have, a reload refetches it instead of leaving the tab empty.
    private var backtestRequested = false

    init(api: APIClient = .shared) {
        self.api = api
    }

    /// Portfolio data is independent of the chart period. Publish each resource
    /// when it arrives, and reject every update belonging to an old selection.
    func reload(currency: String, accounts: [String]?, period: Period, showClosed: Bool, benchmarks: [String],
                customFrom: String? = nil, customTo: String? = nil,
                visibleItems: Set<String> = TabLayout.performanceInitialVisible, force: Bool = false) {
        loadTask?.cancel()
        generation = UUID()
        let revision = generation
        let scope = "\(APIConfig.baseURL)|\(currency)|\((accounts ?? []).sorted())|\(showClosed)|\(visibleItems.sorted())"
        let sameSelection = loadedScope == scope
        let sameScope = sameSelection && !force
        if !sameSelection {
            loadedScope = nil
            metrics = nil; holdings = []; risk = nil; projection = nil; health = nil
            attribution = nil; dividendEvents = []; earningsEvents = []
            heroHistory = []; heroIntraday = []; heroWTD = []; calculatedAt = nil; backendRefreshing = false
            projectionBacktest = nil; backtestKey = nil; isLoadingBacktest = false; chartCache.removeAll()
            history = []
        }
        widgetErrors = [:]; errorMessage = nil; isLoading = true
        loadTask = Task { [weak self] in
            guard let self else { return }
            defer { if self.generation == revision { self.isLoading = false } }
            let cur = URLQueryItem(name: "currency", value: currency)
            let accountItems = APIClient.arrayQuery("accounts", accounts)
            let core = [cur, URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false")] + accountItems
            let base = [cur] + accountItems
            var chart = base + [URLQueryItem(name: "period", value: period.rawValue), URLQueryItem(name: "interval", value: period.interval)] + APIClient.arrayQuery("benchmarks", benchmarks)
            if period == .custom {
                if let from = customFrom { chart.append(URLQueryItem(name: "from", value: from)) }
                if let to = customTo { chart.append(URLQueryItem(name: "to", value: to)) }
            }
            let chartItems = chart
            let chartKey = chartItems.map { "\($0.name)=\($0.value ?? "")" }.joined(separator: "&")
            if let cached = self.chartCache[chartKey] { self.history = cached }
            else if !force { self.history = [] }
            let needsChart = visibleItems.contains("performanceGraph") && (force || self.chartCache[chartKey] == nil)
            let needsCore = !sameScope
            async let headline: Void = self.fetch("/summary/headline", query: base, revision: revision, enabled: needsCore) { (response: SummaryResponse) in
                if self.metrics == nil { self.metrics = response.metrics; self.calculatedAt = response.freshness?.date; self.backendRefreshing = response.freshness?.refreshing ?? false }
            }
            async let summary: Void = self.fetch("/summary", query: core, revision: revision, enabled: needsCore, critical: true) { (response: SummaryResponse) in
                self.metrics = response.metrics; self.calculatedAt = response.freshness?.date; self.backendRefreshing = response.freshness?.refreshing ?? false
            }
            async let holdings: Void = self.fetch("/holdings", query: core, revision: revision, enabled: needsCore, critical: true) { (value: [Holding]) in self.holdings = value }
            async let history: Void = self.fetch("/history", query: chartItems, revision: revision, enabled: needsChart, critical: true) { (value: [PerformancePoint]) in
                self.history = value
                if self.chartCache.count >= 10 { self.chartCache.removeAll() }
                self.chartCache[chartKey] = value
            }
            let needsHero = needsCore && visibleItems.contains("portfolioHero")
            async let heroHistory: Void = self.fetch("/history", query: base + [URLQueryItem(name: "period", value: "1y"), URLQueryItem(name: "interval", value: "1d")], revision: revision, enabled: needsHero) { (value: [PerformancePoint]) in self.heroHistory = value }
            async let heroIntraday: Void = self.fetch("/history", query: base + [URLQueryItem(name: "period", value: "1d"), URLQueryItem(name: "interval", value: "5m")], revision: revision, enabled: needsHero) { (value: [PerformancePoint]) in self.heroIntraday = value }
            async let heroWTD: Void = self.fetch("/history", query: base + [URLQueryItem(name: "period", value: "5d"), URLQueryItem(name: "interval", value: "15m")], revision: revision, enabled: needsHero) { (value: [PerformancePoint]) in self.heroWTD = value }
            // YTD is supplied by risk metrics even when Risk Analytics is hidden.
            async let risk: Void = self.fetch("/risk_metrics", query: core, revision: revision, enabled: needsCore && !visibleItems.isDisjoint(with: ["riskMetrics", "ytdReturn"])) { (value: RiskMetrics) in self.risk = value }
            async let projection: Void = self.fetch("/projection", query: base, revision: revision, enabled: needsCore && visibleItems.contains("projection")) { (value: Projection) in self.projection = value }
            async let health: Void = self.fetch("/portfolio_health", query: core, revision: revision, enabled: needsCore && !visibleItems.isDisjoint(with: ["riskMetrics", "dashboardInsights"])) { (value: PortfolioHealth) in self.health = value }
            async let attr: Void = self.fetch("/attribution", query: core, revision: revision, enabled: needsCore && !visibleItems.isDisjoint(with: ["sectorContribution", "topContributors"])) { (value: Attribution) in self.attribution = value }
            async let dividends: Void = self.fetch("/dividend_calendar", query: base, revision: revision, enabled: needsCore && visibleItems.contains("dashboardEvents")) { (value: [DividendEvent]) in self.dividendEvents = value }
            async let earnings: Void = self.fetch("/earnings_calendar", query: accountItems, revision: revision, enabled: needsCore && visibleItems.contains("dashboardEvents")) { (value: [EarningsEvent]) in self.earningsEvents = value }
            _ = await (headline, summary, holdings, history, heroHistory, heroIntraday, heroWTD, risk, projection, health, attr, dividends, earnings)
            guard self.generation == revision, !Task.isCancelled else { return }
            if self.errorMessage == nil { self.loadedScope = scope }
            if self.backtestRequested && needsCore { await self.loadProjectionBacktest(currency: currency, accounts: accounts) }
            if self.backendRefreshing {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard self.generation == revision, !Task.isCancelled else { return }
                self.reload(currency: currency, accounts: accounts, period: period, showClosed: showClosed,
                            benchmarks: benchmarks, customFrom: customFrom, customTo: customTo,
                            visibleItems: visibleItems, force: true)
            }
        }
    }

    private func fetch<T: Decodable & Sendable>(_ path: String, query: [URLQueryItem], revision: UUID,
                                      enabled: Bool, critical: Bool = false, publish: @MainActor (T) -> Void) async {
        guard enabled, generation == revision, !Task.isCancelled else { return }
        do {
            let value: T = try await api.get(path, query: query)
            guard generation == revision, !Task.isCancelled else { return }
            publish(value)
        } catch {
            guard generation == revision, !Task.isCancelled else { return }
            widgetErrors[path] = error.localizedDescription
            if critical { errorMessage = "Could not load portfolio data. Retry to update it." }
        }
    }

    /// Fetch the projection backtest for the current selection, once. The walk-
    /// forward refit is expensive, so this runs only when the Backtest tab is open.
    func loadProjectionBacktest(currency: String, accounts: [String]?) async {
        backtestRequested = true
        let key = "\(currency)|\((accounts ?? []).joined(separator: ","))"
        if backtestKey == key, projectionBacktest != nil { return }
        if isLoadingBacktest { return }
        let revision = generation
        isLoadingBacktest = true
        defer { if generation == revision { isLoadingBacktest = false } }
        let query = [URLQueryItem(name: "currency", value: currency)] + APIClient.arrayQuery("accounts", accounts)
        let result: ProjectionBacktest? = try? await api.get("/projection/backtest", query: query)
        guard generation == revision, !Task.isCancelled else { return }
        // A failed request still has to resolve the card's spinner, so it lands
        // as an explicit unavailable-with-reason rather than staying nil.
        projectionBacktest = result ?? ProjectionBacktest(
            available: false, reason: "error", currency: currency, marketTimezone: nil,
            historyYears: nil, historyStart: nil, historyEnd: nil, requiredYears: nil,
            horizons: nil, replay: nil)
        backtestKey = result == nil ? nil : key
    }
}
