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
}

// MARK: - Price history (`GET /api/stock_history/{symbol}`)

struct StockHistoryPoint: Codable, Sendable, Identifiable {
    let date: String
    let value: Double
    let returnPct: Double?

    enum CodingKeys: String, CodingKey {
        case date, value
        case returnPct = "return_pct"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        date = try c.decodeIfPresent(String.self, forKey: .date) ?? ""
        value = try c.decodeIfPresent(Double.self, forKey: .value) ?? 0
        returnPct = try c.decodeIfPresent(Double.self, forKey: .returnPct)
    }

    var id: String { date }
    var parsedDate: Date? { StockHistoryPoint.fmt.date(from: String(date.prefix(10))) }
    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
}

// MARK: - Intrinsic value (`GET /api/intrinsic_value/{symbol}`)

struct IntrinsicValueResponse: Codable, Sendable {
    let currentPrice: Double?
    let averageIntrinsicValue: Double?
    let marginOfSafetyPct: Double?
    let valuationNote: String?
    let models: Models?

    enum CodingKeys: String, CodingKey {
        case currentPrice = "current_price"
        case averageIntrinsicValue = "average_intrinsic_value"
        case marginOfSafetyPct = "margin_of_safety_pct"
        case valuationNote = "valuation_note"
        case models
    }

    struct Model: Codable, Sendable {
        let intrinsicValue: Double?
        let error: String?
        let model: String?
        enum CodingKeys: String, CodingKey {
            case intrinsicValue = "intrinsic_value"
            case error, model
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

    enum CodingKeys: String, CodingKey {
        case financials, cashflow
        case balanceSheet = "balance_sheet"
    }
}

// MARK: - Ratios (`GET /api/ratios/{symbol}`)

struct RatiosResponse: Codable, Sendable {
    let valuation: [String: JSONValue]?
}

// MARK: - AI analysis (`GET /api/stock-analysis/{symbol}`)

struct StockAnalysis: Codable, Sendable {
    let summary: String?
    let aiReview: String?
    let sentiment: Double?
    let scorecard: Scorecard?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case summary, sentiment, scorecard, error
        case aiReview = "ai_review"
    }

    struct Scorecard: Codable, Sendable {
        let moat: Double?
        let financialStrength: Double?
        let predictability: Double?
        let growth: Double?
        enum CodingKeys: String, CodingKey {
            case moat, predictability, growth
            case financialStrength = "financial_strength"
        }
    }
}
