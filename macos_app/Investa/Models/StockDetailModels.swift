import Foundation

// MARK: - Fundamentals (`GET /api/fundamentals/{symbol}`)

/// Wide, partly-dynamic company info. Kept as a raw map with typed accessors.
struct Fundamentals: Codable, Sendable {
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(raw)
    }

    func double(_ k: String) -> Double? { raw[k]?.doubleValue }
    func string(_ k: String) -> String? { raw[k]?.stringValue }

    var name: String? { string("longName") ?? string("shortName") }
    var summary: String? { string("longBusinessSummary") }
    var sector: String? { string("sector") }
    var industry: String? { string("industry") }
    var website: String? { string("website") }
    var currency: String? { string("currency") }
    var exchange: String? { string("exchange") }
    var price: Double? { double("regularMarketPrice") }
    var marketCap: Double? { double("marketCap") }
    var trailingPE: Double? { double("trailingPE") }
    var forwardPE: Double? { double("forwardPE") }
    /// Percent, e.g. 15.0 for a 15% yield. Yahoo's encoding is settled against
    /// rate/price rather than guessed from magnitude — see `DividendYield`.
    var dividendYield: Double? {
        DividendYield.normalize(
            rawYield: double("dividendYield"),
            dividendRate: double("dividendRate") ?? double("trailingAnnualDividendRate"),
            price: double("currentPrice") ?? double("regularMarketPrice"),
            trailingYield: double("trailingAnnualDividendYield")
        )
    }
    var beta: Double? { double("beta") }
    var high52: Double? { double("fiftyTwoWeekHigh") }
    var low52: Double? { double("fiftyTwoWeekLow") }
    var shortName: String? { string("shortName") ?? string("longName") }
    /// Percent, e.g. 0.55 for a 0.55% fee.
    var expenseRatio: Double? {
        DividendYield.normalizeExpenseRatio(
            double("netExpenseRatio") ?? double("expenseRatio") ?? double("annualReportExpenseRatio")
        )
    }
    var isETF: Bool { raw["etf_data"]?.objectValue != nil }

    /// ETF top holdings (symbol, name, percent).
    var etfTopHoldings: [(symbol: String, name: String, percent: Double)] {
        (raw["etf_data"]?["top_holdings"]?.arrayValue ?? []).compactMap { v in
            guard let o = v.objectValue else { return nil }
            return (o["symbol"]?.stringValue ?? "", o["name"]?.stringValue ?? "", o["percent"]?.doubleValue ?? 0)
        }
    }
    var etfSectorWeightings: [(String, Double)] {
        (raw["etf_data"]?["sector_weightings"]?.objectValue ?? [:]).compactMap { k, v in
            v.doubleValue.map { (k, $0) }
        }.sorted { $0.1 > $1.1 }
    }

    /// Valuation / earnings / profitability / market readings, derived
    /// server-side by the same code that builds the heatmap payload — so the
    /// keys are the heatmap's, and `StockMetric` reads both.
    var keyMetrics: [String: Double] {
        (raw["key_metrics"]?.objectValue ?? [:]).compactMapValues { $0.doubleValue }
    }

    /// Next scheduled report, derived server-side (see `server/calendar_events.py`).
    var upcomingEarnings: UpcomingEarnings? {
        UpcomingEarnings(json: raw["upcoming_events"]?["earnings"])
    }
    /// The quarter just reported (within the last few days), with what it printed.
    var recentEarnings: UpcomingEarnings? {
        UpcomingEarnings(json: raw["upcoming_events"]?["recent_earnings"])
    }
    /// Next dividend, per share and in the stock's own currency.
    var upcomingDividend: UpcomingDividend? {
        UpcomingDividend(json: raw["upcoming_events"]?["dividend"])
    }
}

/// An earnings report for one symbol — announced, projected, or (under
/// `status == "reported"`) already printed.
struct UpcomingEarnings: Sendable {
    let date: String
    let dateEnd: String?
    let status: String
    let epsEstimate: Double?
    let epsYearAgo: Double?
    /// What was actually printed; nil until Yahoo attaches the figure.
    let epsActual: Double?
    /// Beat/miss against consensus, in percent.
    let surprisePct: Double?
    /// IANA zone of the reporting exchange — the day count is measured against it
    /// rather than the device clock (see `MarketTime`).
    let marketTimezone: String?

    init?(json: JSONValue?) {
        guard let date = json?["earnings_date"]?.stringValue, !date.isEmpty else { return nil }
        self.date = date
        dateEnd = json?["earnings_date_end"]?.stringValue
        status = json?["status"]?.stringValue ?? "estimated"
        epsEstimate = json?["eps_estimate"]?.doubleValue
        epsYearAgo = json?["eps_year_ago"]?.doubleValue
        epsActual = json?["eps_actual"]?.doubleValue
        surprisePct = json?["surprise_pct"]?.doubleValue
        marketTimezone = json?["market_timezone"]?.stringValue
    }
}

/// An announced or projected dividend payment for one symbol.
struct UpcomingDividend: Sendable {
    let date: String
    let exDate: String?
    let amountPerShare: Double?
    let status: String
    /// IANA zone of the paying exchange — see `UpcomingEarnings.marketTimezone`.
    let marketTimezone: String?

    init?(json: JSONValue?) {
        guard let date = json?["dividend_date"]?.stringValue, !date.isEmpty else { return nil }
        self.date = date
        exDate = json?["ex_dividend_date"]?.stringValue
        amountPerShare = json?["amount_per_share"]?.doubleValue
        status = json?["status"]?.stringValue ?? "estimated"
        marketTimezone = json?["market_timezone"]?.stringValue
    }
}

// MARK: - Price history (`GET /api/stock_history/{symbol}`)

struct StockHistoryPoint: Codable, Sendable, Identifiable {
    let date: String
    let value: Double
    let returnPct: Double?
    let volume: Double
    /// Benchmark return-% columns keyed by Yahoo ticker (e.g. `^GSPC`).
    let benchmarks: [String: Double]

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        date = raw["date"]?.stringValue ?? ""
        value = raw["value"]?.doubleValue ?? 0
        returnPct = raw["return_pct"]?.doubleValue
        volume = raw["volume"]?.doubleValue ?? 0
        var bench: [String: Double] = [:]
        for (k, v) in raw where k.hasPrefix("^") { if let d = v.doubleValue { bench[k] = d } }
        benchmarks = bench
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CK.self)
        try c.encode(date, forKey: .date); try c.encode(value, forKey: .value)
    }
    private enum CK: String, CodingKey { case date, value }

    var id: String { date }

    /// Parses either a date (`yyyy-MM-dd`) or an intraday timestamp.
    var parsedDate: Date? {
        if let d = StockHistoryPoint.dayFmt.date(from: String(date.prefix(10))),
           date.count <= 10 { return d }
        return StockHistoryPoint.isoFmt.date(from: date)
            ?? StockHistoryPoint.isoFmt2.date(from: date)
            ?? StockHistoryPoint.dayFmt.date(from: String(date.prefix(10)))
    }
    private static let dayFmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    private static let isoFmt: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter(); f.formatOptions = [.withInternetDateTime]; return f
    }()
    private static let isoFmt2: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"; return f
    }()
}

// MARK: - Intrinsic value (`GET /api/intrinsic_value/{symbol}`)

struct IntrinsicValueResponse: Codable, Sendable {
    let currentPrice: Double?
    /// Nil when the backend declines to value the company — check `status`.
    let averageIntrinsicValue: Double?
    let marginOfSafetyPct: Double?
    let valuationNote: String?
    let valuationStatus: String?
    /// Spread between contributing models, as % of the blended value.
    let modelSpreadPct: Double?
    /// Value of current earning power assuming zero growth.
    let earningsPowerFloor: Double?
    /// Peter Lynch Fair Value estimate.
    let lynchFairValue: Double?
    let recommendedMethod: RecommendedMethod?
    let models: Models?
    let range: Range?

    enum CodingKeys: String, CodingKey {
        case currentPrice = "current_price"
        case averageIntrinsicValue = "average_intrinsic_value"
        case marginOfSafetyPct = "margin_of_safety_pct"
        case valuationNote = "valuation_note"
        case valuationStatus = "valuation_status"
        case modelSpreadPct = "model_spread_pct"
        case earningsPowerFloor = "earnings_power_floor"
        case lynchFairValue = "lynch_fair_value"
        case recommendedMethod = "recommended_method"
        case models, range
    }

    /// Why the response looks the way it does. Mirrors `ValuationStatus` in
    /// `web_app/lib/api.ts`; keep the two in step.
    enum Status: String, Sendable {
        case ok, lowConfidence = "low_confidence", clamped
        case ineligible, noModel = "no_model", nav
    }

    var status: Status? { valuationStatus.flatMap(Status.init(rawValue:)) }

    /// True when the backend refused to produce a value at all.
    var isRefusal: Bool { status == .ineligible || status == .noModel }

    struct RecommendedMethod: Codable, Sendable {
        let methodKey: String?
        let name: String?
        let bestSuitedFor: String?
        let keyCaveats: String?
        let whenToUse: String?
        let keyLimitation: String?
        let rationale: String?
        let intrinsicValue: Double?

        enum CodingKeys: String, CodingKey {
            case methodKey = "method_key"
            case name
            case bestSuitedFor = "best_suited_for"
            case keyCaveats = "key_caveats"
            case whenToUse = "when_to_use"
            case keyLimitation = "key_limitation"
            case rationale
            case intrinsicValue = "intrinsic_value"
        }
    }

    struct Range: Codable, Sendable { let bear: Double?; let bull: Double? }
    struct HistogramPoint: Codable, Sendable { let price: Double?; let count: Double? }
    struct MC: Codable, Sendable { let bear: Double?; let base: Double?; let bull: Double?; let histogram: [HistogramPoint]? }
    struct Model: Codable, Sendable {
        let intrinsicValue: Double?
        let error: String?
        let model: String?
        let mc: MC?
        let parameters: [String: JSONValue]?
        enum CodingKeys: String, CodingKey {
            case intrinsicValue = "intrinsic_value"
            case error, model, mc, parameters
        }
    }
    struct Models: Codable, Sendable {
        let dcf: Model?
        let dcfo: Model?
        let dni: Model?
        let meanPe: Model?
        let peg: Model?
        let meanPb: Model?
        let meanPs: Model?
        let psg: Model?
        let graham: Model?
        let ddm: Model?
        let epv: Model?
        let lynch: Model?

        enum CodingKeys: String, CodingKey {
            case dcf, dcfo, dni
            case meanPe = "mean_pe"
            case peg
            case meanPb = "mean_pb"
            case meanPs = "mean_ps"
            case psg
            case graham, ddm, epv, lynch
        }
    }
}

// MARK: - Earnings (`GET /api/earnings_dates/{symbol}`)

struct EarningsDate: Codable, Sendable, Identifiable {
    let date: String
    let epsEstimate: Double?
    let epsActual: Double?
    let surprisePct: Double?

    enum CodingKeys: String, CodingKey {
        case date
        case epsEstimate = "eps_estimate"
        case epsActual = "eps_actual"
        case surprisePct = "surprise_pct"
    }
    var id: String { date }
}

// MARK: - Financial statements (`GET /api/financials/{symbol}`)

/// A statement as a matrix: `index` are line-item rows, `columns` are periods,
/// `data[row][col]` the values.
struct FinancialStatement: Codable, Sendable {
    let columns: [String]
    let index: [String]
    let data: [[Double?]]

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        columns = try c.decodeIfPresent([String].self, forKey: .columns) ?? []
        index = try c.decodeIfPresent([String].self, forKey: .index) ?? []
        data = try c.decodeIfPresent([[Double?]].self, forKey: .data) ?? []
    }
    enum CodingKeys: String, CodingKey { case columns, index, data }
}

struct FinancialsResponse: Codable, Sendable {
    let financials: FinancialStatement?
    let balanceSheet: FinancialStatement?
    let cashflow: FinancialStatement?
    let shareholdersEquity: FinancialStatement?

    enum CodingKeys: String, CodingKey {
        case financials, cashflow
        case balanceSheet = "balance_sheet"
        case shareholdersEquity = "shareholders_equity"
    }
}

// MARK: - Track record (`GET /api/track-record/{symbol}`)

/// One measured metric. `display` is preformatted by the backend so the web,
/// macOS and iOS clients cannot render the same number three ways; `note` says
/// why a metric is unmeasurable when that is knowable (a stock split breaking
/// the share-count series, for one).
struct TrackRecordItem: Decodable, Sendable, Identifiable {
    let key: String
    let label: String
    let unit: String
    let value: Double?
    let display: String?
    let note: String?
    let higherIsBetter: Bool

    var id: String { key }

    enum CodingKeys: String, CodingKey {
        case key, label, unit, value, display, note
        case higherIsBetter = "higher_is_better"
    }
}

struct TrackRecordGroup: Decodable, Sendable, Identifiable {
    let key: String
    let title: String
    let items: [TrackRecordItem]

    var id: String { key }
}

struct TrackRecordRank: Decodable, Sendable {
    let rank: Int?
    let compositeScore: Double?
    let qualityScore: Double?
    let valueScore: Double?
    let confidence: Double?
    let pillars: [String: Double?]?

    enum CodingKeys: String, CodingKey {
        case rank, confidence, pillars
        case compositeScore = "composite_score"
        case qualityScore = "quality_score"
        case valueScore = "value_score"
    }
}

/// Today's multiple against the company's own history. Absent entirely when the
/// local price store is too shallow to support a band.
struct TrackRecordBand: Decodable, Sendable, Identifiable {
    let metric: String
    let label: String
    let current: Double
    let median: Double
    let p25: Double
    let p75: Double
    let low: Double
    let high: Double
    /// 0 = cheapest it has ever been, 100 = dearest.
    let percentile: Double
    let observations: Int
    let display: String
    let medianDisplay: String
    /// "dearer than usual for this company" — a comparison, never advice.
    let summary: String

    var id: String { metric }

    enum CodingKeys: String, CodingKey {
        case metric, label, current, median, p25, p75, low, high, percentile, observations
        case display, summary
        case medianDisplay = "median_display"
    }
}

/// How one metric behaved peak-to-trough in a downturn.
struct TrackRecordStressItem: Decodable, Sendable, Identifiable {
    let metric: String
    let label: String
    let peakYear: Int
    let troughYear: Int
    let changePct: Double
    let display: String
    let recoveryDisplay: String?

    var id: String { metric }

    enum CodingKeys: String, CodingKey {
        case metric, label, display
        case peakYear = "peak_year"
        case troughYear = "trough_year"
        case changePct = "change_pct"
        case recoveryDisplay = "recovery_display"
    }
}

/// One downturn. `covered` is false when the company has no filings spanning it —
/// "not listed then" and "did not fall" are opposite claims and must not render
/// the same way.
struct TrackRecordStress: Decodable, Sendable, Identifiable {
    let key: String
    let label: String
    let covered: Bool
    let items: [TrackRecordStressItem]

    var id: String { key }
}

/// A number the company changed after first reporting it. `display` is
/// preformatted by the backend ("$1.95bn → $4.41bn") so the three clients cannot
/// render the same revision three ways.
struct TrackRecordRevision: Decodable, Sendable, Identifiable {
    let concept: String
    let label: String
    let periodEnd: String
    let changePct: Double
    let display: String
    let changeDisplay: String
    let firstFiled: String
    let restatedFiled: String

    var id: String { "\(concept)-\(periodEnd)" }

    enum CodingKeys: String, CodingKey {
        case concept, label, display
        case periodEnd = "period_end"
        case changePct = "change_pct"
        case changeDisplay = "change_display"
        case firstFiled = "first_filed"
        case restatedFiled = "restated_filed"
    }
}

struct TrackRecordRevisions: Decodable, Sendable {
    let count: Int
    let items: [TrackRecordRevision]
}

/// The measured quality record: the metrics the Buffett ranking scores on, over
/// the durability window, with the span of filings they rest on.
struct TrackRecord: Decodable, Sendable {
    let symbol: String
    let name: String?
    let model: String
    let periodCount: Int
    let firstPeriod: String?
    let latestPeriod: String?
    let windowYears: Int
    let coverage: Double?
    let gateFailures: [String]
    let rank: TrackRecordRank?
    let groups: [TrackRecordGroup]
    let revisions: TrackRecordRevisions?
    let stress: [TrackRecordStress]?
    let valuationBands: [TrackRecordBand]?

    enum CodingKeys: String, CodingKey {
        case symbol, name, model, coverage, rank, groups, revisions, stress
        case valuationBands = "valuation_bands"
        case periodCount = "period_count"
        case firstPeriod = "first_period"
        case latestPeriod = "latest_period"
        case windowYears = "window_years"
        case gateFailures = "gate_failures"
    }
}

// MARK: - Ratios (`GET /api/ratios/{symbol}`)

struct RatiosResponse: Decodable, Sendable {
    let valuation: [String: JSONValue]?
    /// Historical ratio rows (each has a `Period` plus dynamic metric keys).
    let historical: [[String: JSONValue]]

    enum CodingKeys: String, CodingKey {
        case valuation, historical
    }

    init(valuation: [String: JSONValue]? = nil, historical: [[String: JSONValue]] = []) {
        self.valuation = valuation
        self.historical = historical
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.valuation = try container.decodeIfPresent([String: JSONValue].self, forKey: .valuation)
        self.historical = try container.decodeIfPresent([[String: JSONValue]].self, forKey: .historical) ?? []
    }
}

// MARK: - AI analysis (`GET /api/stock-analysis/{symbol}`)

struct StockAnalysis: Decodable, Sendable {
    let summary: String?
    let aiReview: String?
    let sentiment: Double?
    let scorecard: Scorecard?
    let analysis: Analysis?
    let catalysts: [Catalyst]
    let error: String?

    struct Scorecard: Sendable {
        let moat: Double?; let financialStrength: Double?; let predictability: Double?; let growth: Double?
    }
    struct Analysis: Sendable {
        let moat: String?; let financialStrength: String?; let predictability: String?; let growthPerspective: String?
    }
    struct Catalyst: Sendable, Identifiable { let id = UUID(); let event: String; let date: String; let impact: String }

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        summary = raw["summary"]?.stringValue
        aiReview = raw["ai_review"]?.stringValue
        sentiment = raw["sentiment"]?.doubleValue
        error = raw["error"]?.stringValue
        if let sc = raw["scorecard"]?.objectValue {
            scorecard = Scorecard(moat: sc["moat"]?.doubleValue, financialStrength: sc["financial_strength"]?.doubleValue,
                                  predictability: sc["predictability"]?.doubleValue, growth: sc["growth"]?.doubleValue)
        } else { scorecard = nil }
        if let a = raw["analysis"]?.objectValue {
            analysis = Analysis(moat: a["moat"]?.stringValue, financialStrength: a["financial_strength"]?.stringValue,
                                predictability: a["predictability"]?.stringValue, growthPerspective: a["growth_perspective"]?.stringValue)
        } else { analysis = nil }
        catalysts = (raw["catalysts"]?.arrayValue ?? []).compactMap { v in
            guard let o = v.objectValue else { return nil }
            return Catalyst(event: o["event"]?.stringValue ?? "", date: o["date"]?.stringValue ?? "", impact: o["impact"]?.stringValue ?? "")
        }
    }
}

// MARK: - Single Stock Position & Lots (`GET /api/stock/{symbol}/position`)

public struct OpenLot: Decodable, Sendable, Identifiable {
    public let lotId: Int
    public let date: String
    public let account: String
    public let quantity: Double
    public let costPerShareLocal: Double
    public let costBasisDisplay: Double
    public let marketValueDisplay: Double
    public let unrealizedGainDisplay: Double
    public let unrealizedGainPct: Double
    public let holdingPeriodDays: Int
    public let taxTerm: String

    public var id: String { "\(account)-\(lotId)-\(date)" }

    enum CodingKeys: String, CodingKey {
        case lotId = "lot_id"
        case date, account, quantity
        case costPerShareLocal = "cost_per_share_local"
        case costBasisDisplay = "cost_basis_display"
        case marketValueDisplay = "market_value_display"
        case unrealizedGainDisplay = "unrealized_gain_display"
        case unrealizedGainPct = "unrealized_gain_pct"
        case holdingPeriodDays = "holding_period_days"
        case taxTerm = "tax_term"
    }
}

public struct ClosedTrade: Decodable, Sendable, Identifiable {
    public let sellDate: String
    public let account: String
    public let quantitySold: Double
    public let salePrice: Double
    public let proceedsDisplay: Double
    public let costBasisDisplay: Double
    public let realizedGainDisplay: Double
    public let originalTxId: Int?

    public var id: String { "\(account)-\(sellDate)-\(originalTxId ?? 0)" }

    enum CodingKeys: String, CodingKey {
        case sellDate = "sell_date"
        case account
        case quantitySold = "quantity_sold"
        case salePrice = "sale_price"
        case proceedsDisplay = "proceeds_display"
        case costBasisDisplay = "cost_basis_display"
        case realizedGainDisplay = "realized_gain_display"
        case originalTxId = "original_tx_id"
    }
}

public struct StockPositionSummary: Decodable, Sendable {
    public let quantity: Double
    public let currentPrice: Double
    public let marketValue: Double
    public let avgCostPrice: Double
    public let costBasis: Double
    public let totalBuyCost: Double
    public let portfolioWeightPct: Double?

    enum CodingKeys: String, CodingKey {
        case quantity
        case currentPrice = "current_price"
        case marketValue = "market_value"
        case avgCostPrice = "avg_cost_price"
        case costBasis = "cost_basis"
        case totalBuyCost = "total_buy_cost"
        case portfolioWeightPct = "portfolio_weight_pct"
    }
}

public struct StockReturnAttribution: Decodable, Sendable {
    public let unrealizedGain: Double
    public let unrealizedGainPct: Double
    public let realizedGain: Double
    public let lifetimeDividends: Double
    public let commissions: Double
    public let withholdingTaxes: Double
    public let totalGain: Double
    public let totalReturnPct: Double
    public let irrPct: Double?
    public let twrrPct: Double?
    public let indicatedAnnualDividend: Double
    public let yieldOnCostPct: Double?
    public let marketYieldPct: Double?
    public let fxGainLoss: Double
    public let fxGainLossPct: Double

    enum CodingKeys: String, CodingKey {
        case unrealizedGain = "unrealized_gain"
        case unrealizedGainPct = "unrealized_gain_pct"
        case realizedGain = "realized_gain"
        case lifetimeDividends = "lifetime_dividends"
        case commissions
        case withholdingTaxes = "withholding_taxes"
        case totalGain = "total_gain"
        case totalReturnPct = "total_return_pct"
        case irrPct = "irr_pct"
        case twrrPct = "twrr_pct"
        case indicatedAnnualDividend = "indicated_annual_dividend"
        case yieldOnCostPct = "yield_on_cost_pct"
        case marketYieldPct = "market_yield_pct"
        case fxGainLoss = "fx_gain_loss"
        case fxGainLossPct = "fx_gain_loss_pct"
    }
}

public struct StockPositionResponse: Decodable, Sendable {
    public let symbol: String
    public let displayCurrency: String
    public let localCurrency: String
    public let fxRate: Double
    public let hasPosition: Bool
    public let summary: StockPositionSummary?
    public let returns: StockReturnAttribution?
    public let openLots: [OpenLot]
    public let closedTrades: [ClosedTrade]

    enum CodingKeys: String, CodingKey {
        case symbol
        case displayCurrency = "display_currency"
        case localCurrency = "local_currency"
        case fxRate = "fx_rate"
        case hasPosition = "has_position"
        case summary, returns
        case openLots = "open_lots"
        case closedTrades = "closed_trades"
    }
}

public struct StockPositionHistoryPoint: Codable, Sendable, Identifiable {
    public let date: String
    public let value: Double
    public let costBasis: Double
    public let shares: Double
    public let unrealizedGain: Double
    public let unrealizedGainPct: Double
    public let returnPct: Double
    public let benchmarks: [String: Double]

    public var id: String { date }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        date = raw["date"]?.stringValue ?? ""
        value = raw["value"]?.doubleValue ?? 0
        costBasis = raw["cost_basis"]?.doubleValue ?? 0
        shares = raw["shares"]?.doubleValue ?? 0
        unrealizedGain = raw["unrealized_gain"]?.doubleValue ?? 0
        unrealizedGainPct = raw["unrealized_gain_pct"]?.doubleValue ?? 0
        returnPct = raw["return_pct"]?.doubleValue ?? 0

        var bench: [String: Double] = [:]
        let known = Set(["date", "value", "cost_basis", "shares", "unrealized_gain", "unrealized_gain_pct", "return_pct"])
        for (k, v) in raw where !known.contains(k) {
            if let d = v.doubleValue { bench[k] = d }
        }
        benchmarks = bench
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CK.self)
        try c.encode(date, forKey: .date)
        try c.encode(value, forKey: .value)
        try c.encode(costBasis, forKey: .costBasis)
        try c.encode(shares, forKey: .shares)
        try c.encode(unrealizedGain, forKey: .unrealizedGain)
        try c.encode(unrealizedGainPct, forKey: .unrealizedGainPct)
        try c.encode(returnPct, forKey: .returnPct)
    }

    private enum CK: String, CodingKey {
        case date, value
        case costBasis = "cost_basis"
        case shares
        case unrealizedGain = "unrealized_gain"
        case unrealizedGainPct = "unrealized_gain_pct"
        case returnPct = "return_pct"
    }

    public var parsedDate: Date? {
        StockPositionHistoryPoint.dayFmt.date(from: String(date.prefix(10)))
    }
    private static let dayFmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
}

