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
    var dividendYield: Double? { double("dividendYield") }
    var beta: Double? { double("beta") }
    var high52: Double? { double("fiftyTwoWeekHigh") }
    var low52: Double? { double("fiftyTwoWeekLow") }
    var shortName: String? { string("shortName") ?? string("longName") }
    var expenseRatio: Double? { double("netExpenseRatio") ?? double("expenseRatio") ?? double("annualReportExpenseRatio") }
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
    let averageIntrinsicValue: Double?
    let marginOfSafetyPct: Double?
    let valuationNote: String?
    let models: Models?
    let range: Range?

    enum CodingKeys: String, CodingKey {
        case currentPrice = "current_price"
        case averageIntrinsicValue = "average_intrinsic_value"
        case marginOfSafetyPct = "margin_of_safety_pct"
        case valuationNote = "valuation_note"
        case models, range
    }

    struct Range: Codable, Sendable { let bear: Double?; let bull: Double? }
    struct MC: Codable, Sendable { let bear: Double?; let base: Double?; let bull: Double? }
    struct Model: Codable, Sendable {
        let intrinsicValue: Double?
        let error: String?
        let model: String?
        let mc: MC?
        enum CodingKeys: String, CodingKey {
            case intrinsicValue = "intrinsic_value"
            case error, model, mc
        }
    }
    struct Models: Codable, Sendable {
        let dcf: Model?
        let graham: Model?
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

// MARK: - Ratios (`GET /api/ratios/{symbol}`)

struct RatiosResponse: Decodable, Sendable {
    let valuation: [String: JSONValue]?
    /// Historical ratio rows (each has a `Period` plus dynamic metric keys).
    let historical: [[String: JSONValue]]

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        valuation = raw["valuation"]?.objectValue
        historical = (raw["historical"]?.arrayValue ?? []).compactMap { $0.objectValue }
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
