import Foundation

@MainActor
final class StockDetailViewModel: ObservableObject {
    let symbol: String

    @Published var fundamentals: Fundamentals?
    @Published var history: [StockHistoryPoint] = []
    @Published var intrinsic: IntrinsicValueResponse?
    @Published var earnings: [EarningsDate] = []
    @Published var analysis: StockAnalysis?
    @Published var financials: FinancialsResponse?
    @Published var ratios: RatiosResponse?
    @Published var userPosition: Holding?
    @Published var news: [MarketNewsItem] = []

    @Published var isLoading = false
    @Published var isLoadingAnalysis = false
    @Published var isLoadingFinancials = false
    @Published var isLoadingNews = false
    @Published var period = "1y"
    @Published var errorMessage: String?

    let currency: String
    private let api: APIClient

    init(symbol: String, currency: String = "USD", api: APIClient = .shared) {
        self.symbol = symbol
        self.currency = currency
        self.api = api
    }

    func loadAll() async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        async let f: Fundamentals = api.get("/fundamentals/\(symbol)")
        async let iv: IntrinsicValueResponse = api.get("/intrinsic_value/\(symbol)")
        async let e: [EarningsDate] = api.get("/earnings_dates/\(symbol)")
        async let h: [Holding] = api.get("/holdings", query: [URLQueryItem(name: "currency", value: currency)])
        do { fundamentals = try await f } catch { errorMessage = (error as? APIError)?.errorDescription }
        do { intrinsic = try await iv } catch {}
        do { earnings = try await e } catch {}
        // Aggregate the user's position in this symbol across accounts.
        if let holdings = try? await h {
            userPosition = aggregatePosition(holdings.filter { $0.symbol == symbol })
        }
    }

    private func aggregatePosition(_ rows: [Holding]) -> Holding? {
        rows.first   // backend already aggregates per symbol+account; first match is representative
    }

    func loadNews() async {
        guard news.isEmpty else { return }
        isLoadingNews = true
        defer { isLoadingNews = false }
        news = (try? await api.get("/markets/news",
            query: [URLQueryItem(name: "symbols", value: symbol), URLQueryItem(name: "limit", value: "20")])) ?? []
    }

    func loadHistory() async {
        do {
            history = try await api.get(
                "/stock_history/\(symbol)",
                query: [
                    URLQueryItem(name: "period", value: period),
                    URLQueryItem(name: "interval", value: "1d"),
                ]
            )
        } catch { /* tolerate */ }
    }

    /// Financial statements + ratios are heavy, so loaded on demand.
    func loadFinancials() async {
        guard financials == nil else { return }
        isLoadingFinancials = true
        defer { isLoadingFinancials = false }
        async let fin: FinancialsResponse = api.get(
            "/financials/\(symbol)", query: [URLQueryItem(name: "period_type", value: "annual")]
        )
        async let rat: RatiosResponse = api.get("/ratios/\(symbol)")
        do { financials = try await fin } catch {}
        do { ratios = try await rat } catch {}
    }

    /// AI analysis is expensive, so it's loaded on demand.
    func loadAnalysis(force: Bool = false) async {
        isLoadingAnalysis = true
        defer { isLoadingAnalysis = false }
        do {
            analysis = try await api.get(
                "/stock-analysis/\(symbol)",
                query: [URLQueryItem(name: "force", value: force ? "true" : "false")]
            )
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }
}
