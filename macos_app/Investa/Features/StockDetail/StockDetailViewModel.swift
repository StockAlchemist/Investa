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
    /// Quarterly by default: it is the reporting cadence a holder actually
    /// follows. Only the annual statements carry the SEC-filed back history.
    @Published var financialsPeriod: StatementPeriod = .quarterly
    @Published var ratios: RatiosResponse?
    /// Ratios follow the same cadence rule as the statements: quarterly first.
    @Published var ratiosPeriod: StatementPeriod = .quarterly
    @Published var trackRecord: TrackRecord?
    @Published var userPosition: Holding?
    @Published var news: [MarketNewsItem] = []

    @Published var isLoading = false
    @Published var isLoadingAnalysis = false
    @Published var isLoadingFinancials = false
    @Published var isLoadingRatios = false
    @Published var isLoadingNews = false
    @Published var period = "1y"
    @Published var errorMessage: String?

    let currency: String
    private let api: APIClient
    private var statementCache: [StatementPeriod: FinancialsResponse] = [:]
    private var ratioCache: [StatementPeriod: RatiosResponse] = [:]
    /// Whether the track record has been asked for, as opposed to found. A 404 is
    /// the normal answer for anything that does not file with the SEC, so keying
    /// the retry off `trackRecord == nil` would ask again on every tab switch and
    /// never stop for exactly the holdings that can never have one.
    private var trackRecordRequested = false

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
        do { intrinsic = try await iv } catch { print("Intrinsic error: \(error)") }
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

    /// Financial statements + ratios are heavy, so loaded on demand. Statements
    /// are cached per period, so flipping Quarterly/Annual costs one fetch each.
    func loadFinancials(period: StatementPeriod? = nil) async {
        let wanted = period ?? financialsPeriod
        financialsPeriod = wanted

        if let cached = statementCache[wanted] {
            financials = cached
        } else {
            financials = nil
            isLoadingFinancials = true
            do {
                let fin: FinancialsResponse = try await api.get(
                    "/financials/\(symbol)",
                    query: [URLQueryItem(name: "period_type", value: wanted.rawValue)]
                )
                statementCache[wanted] = fin
                // A switch while this was in flight must not overwrite the newer one.
                if financialsPeriod == wanted { financials = fin }
            } catch {}
            isLoadingFinancials = false
        }

        await loadRatios()

        if !trackRecordRequested {
            trackRecordRequested = true
            // A 404 here is the normal answer for anything that does not file with
            // the SEC — every SET holding — so the panel simply stays hidden.
            do { trackRecord = try await api.get("/track-record/\(symbol)") } catch {}
        }
    }

    /// The ratio history, cached per period. Quarterly measures the same ratios
    /// on trailing-twelve-month flows at each quarter end, so switching is a
    /// change of sampling rate rather than of measurement.
    func loadRatios(period: StatementPeriod? = nil) async {
        let wanted = period ?? ratiosPeriod
        ratiosPeriod = wanted

        if let cached = ratioCache[wanted] {
            ratios = cached
            return
        }
        ratios = nil
        isLoadingRatios = true
        defer { isLoadingRatios = false }
        do {
            let response: RatiosResponse = try await api.get(
                "/ratios/\(symbol)",
                query: [URLQueryItem(name: "period_type", value: wanted.rawValue)]
            )
            ratioCache[wanted] = response
            // A switch while this was in flight must not overwrite the newer one.
            if ratiosPeriod == wanted { ratios = response }
        } catch {}
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
