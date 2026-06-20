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
    @Published var risk: RiskMetrics?
    @Published var health: PortfolioHealth?
    @Published var attribution: Attribution?
    @Published var dividendEvents: [DividendEvent] = []
    @Published var indices: [IndexQuote] = []

    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var loadTask: Task<Void, Never>?

    init(api: APIClient = .shared) {
        self.api = api
    }

    /// Reload everything for the given selection. Cancels any in-flight load.
    func reload(currency: String, accounts: [String]?, period: Period, showClosed: Bool, benchmarks: [String],
                customFrom: String? = nil, customTo: String? = nil) {
        loadTask?.cancel()
        loadTask = Task { [weak self] in
            await self?.load(currency: currency, accounts: accounts, period: period, showClosed: showClosed,
                             benchmarks: benchmarks, customFrom: customFrom, customTo: customTo)
        }
    }

    private func load(currency: String, accounts: [String]?, period: Period, showClosed: Bool, benchmarks: [String],
                      customFrom: String?, customTo: String?) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        let accountItems = APIClient.arrayQuery("accounts", accounts)
        let closedItem = URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false")
        let curItem = URLQueryItem(name: "currency", value: currency)

        // Headline first for a fast card, then the rest concurrently.
        do {
            let headline: SummaryResponse = try await api.get(
                "/summary/headline", query: [curItem] + accountItems
            )
            if Task.isCancelled { return }
            self.metrics = headline.metrics
        } catch is CancellationError {
            return
        } catch { /* non-fatal */ }

        async let summaryResult: SummaryResponse = api.get(
            "/summary", query: [curItem, closedItem] + accountItems)
        async let holdingsResult: [Holding] = api.get(
            "/holdings", query: [curItem, closedItem] + accountItems)
        var historyQuery: [URLQueryItem] = [curItem,
            URLQueryItem(name: "period", value: period.rawValue),
            URLQueryItem(name: "interval", value: period.interval)]
        if period == .custom {
            if let f = customFrom { historyQuery.append(URLQueryItem(name: "from", value: f)) }
            if let t = customTo { historyQuery.append(URLQueryItem(name: "to", value: t)) }
        }
        historyQuery += accountItems + APIClient.arrayQuery("benchmarks", benchmarks)
        async let historyResult: [PerformancePoint] = api.get("/history", query: historyQuery)
        async let riskResult: RiskMetrics = api.get(
            "/risk_metrics", query: [curItem, closedItem] + accountItems)
        async let healthResult: PortfolioHealth = api.get(
            "/portfolio_health", query: [curItem, closedItem] + accountItems)
        async let attrResult: Attribution = api.get(
            "/attribution", query: [curItem, closedItem] + accountItems)
        async let divCalResult: [DividendEvent] = api.get(
            "/dividend_calendar", query: [curItem] + accountItems)

        // Core data — surface its error to the user.
        do {
            let (summary, holdings, history) = try await (summaryResult, holdingsResult, historyResult)
            if Task.isCancelled { return }
            if let full = summary.metrics { self.metrics = full }
            self.holdings = holdings
            self.history = history
        } catch is CancellationError {
            return
        } catch let error as APIError {
            if case .unauthorized = error { return }
            self.errorMessage = error.errorDescription
        } catch {
            self.errorMessage = error.localizedDescription
        }

        // Secondary widgets — tolerate individual failures.
        risk = try? await riskResult
        health = try? await healthResult
        attribution = try? await attrResult
        dividendEvents = (try? await divCalResult) ?? []

        // Market Today index strip (independent of the selection).
        if let map: [String: IndexQuote] = try? await api.get("/indices") {
            indices = Array(map.values)
        }

        // Hero long-history (1y) for the WTD/MTD/YTD/1Y period selector.
        heroHistory = (try? await api.get(
            "/history",
            query: [curItem, URLQueryItem(name: "period", value: "1y"),
                    URLQueryItem(name: "interval", value: "1d")] + accountItems)) ?? []
    }
}
