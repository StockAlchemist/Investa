import Foundation

/// Loads and holds the dashboard's data: headline metrics, full summary,
/// holdings, and the performance history series. Refetches when the shared
/// selection (currency / accounts / period) changes.
@MainActor
final class DashboardViewModel: ObservableObject {
    @Published var metrics: Metrics?
    @Published var holdings: [Holding] = []
    @Published var history: [PerformancePoint] = []

    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var loadTask: Task<Void, Never>?

    init(api: APIClient = .shared) {
        self.api = api
    }

    /// Reload everything for the given selection. Cancels any in-flight load.
    func reload(currency: String, accounts: [String]?, period: Period, showClosed: Bool) {
        loadTask?.cancel()
        loadTask = Task { [weak self] in
            await self?.load(currency: currency, accounts: accounts,
                             period: period, showClosed: showClosed)
        }
    }

    private func load(currency: String, accounts: [String]?, period: Period, showClosed: Bool) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        let accountItems = APIClient.arrayQuery("accounts", accounts)

        // Headline first for a fast card, then the rest concurrently.
        do {
            let headline: SummaryResponse = try await api.get(
                "/summary/headline",
                query: [URLQueryItem(name: "currency", value: currency)] + accountItems
            )
            if Task.isCancelled { return }
            self.metrics = headline.metrics
        } catch is CancellationError {
            return
        } catch {
            // Headline failure isn't fatal; the full summary below may still work.
        }

        async let summaryResult: SummaryResponse = api.get(
            "/summary",
            query: [
                URLQueryItem(name: "currency", value: currency),
                URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false"),
            ] + accountItems
        )
        async let holdingsResult: [Holding] = api.get(
            "/holdings",
            query: [
                URLQueryItem(name: "currency", value: currency),
                URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false"),
            ] + accountItems
        )
        async let historyResult: [PerformancePoint] = api.get(
            "/history",
            query: [
                URLQueryItem(name: "currency", value: currency),
                URLQueryItem(name: "period", value: period.rawValue),
                URLQueryItem(name: "interval", value: "1d"),
            ] + accountItems
        )

        do {
            let (summary, holdings, history) = try await (summaryResult, holdingsResult, historyResult)
            if Task.isCancelled { return }
            if let fullMetrics = summary.metrics {
                self.metrics = fullMetrics
            }
            self.holdings = holdings
            self.history = history
        } catch is CancellationError {
            return
        } catch let error as APIError {
            if case .unauthorized = error { return } // handled globally
            self.errorMessage = error.errorDescription
        } catch {
            self.errorMessage = error.localizedDescription
        }
    }
}
