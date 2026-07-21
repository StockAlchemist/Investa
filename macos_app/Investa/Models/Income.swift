import Foundation

/// One bar from `GET /api/projected_income`. Carries dynamic per-symbol keys for
/// the stacked breakdown, which we keep in `raw` but don't chart in the MVP.
struct ProjectedIncome: Codable, Sendable, Identifiable {
    let month: String
    let value: Double
    let yearMonth: String
    /// Full row incl. dynamic per-symbol keys for the stacked breakdown.
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        let map = try decoder.singleValueContainer().decode([String: JSONValue].self)
        raw = map
        month = map["month"]?.stringValue ?? ""
        value = map["value"]?.doubleValue ?? 0
        yearMonth = map["year_month"]?.stringValue ?? month
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(raw)
    }

    /// Per-symbol contributions (keys other than month/value/year_month).
    var segments: [(symbol: String, amount: Double)] {
        raw.compactMap { (k, v) in
            guard k != "month", k != "value", k != "year_month", let d = v.doubleValue else { return nil }
            return (k, d)
        }.sorted { $0.symbol < $1.symbol }
    }

    var id: String { yearMonth }
}

/// An upcoming dividend from `GET /api/dividend_calendar`.
struct DividendEvent: Codable, Sendable, Identifiable {
    let symbol: String
    /// Company name; absent for synthetic rows such as cash interest.
    let name: String?
    let dividendDate: String
    let exDividendDate: String
    let amount: Double
    let status: String

    enum CodingKeys: String, CodingKey {
        case symbol, name
        case dividendDate = "dividend_date"
        case exDividendDate = "ex_dividend_date"
        case amount, status
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? "?"
        name = try c.decodeIfPresent(String.self, forKey: .name)
        dividendDate = try c.decodeIfPresent(String.self, forKey: .dividendDate) ?? ""
        exDividendDate = try c.decodeIfPresent(String.self, forKey: .exDividendDate) ?? ""
        amount = try c.decodeIfPresent(Double.self, forKey: .amount) ?? 0
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? "estimated"
    }

    var id: String { "\(symbol)|\(exDividendDate)" }
}

/// An upcoming earnings report from `GET /api/earnings_calendar`.
struct EarningsEvent: Codable, Sendable, Identifiable {
    let symbol: String
    let name: String?
    let earningsDate: String
    /// Set only when the company announced a window rather than an exact day.
    let earningsDateEnd: String?
    let status: String
    let epsEstimate: Double?
    let epsYearAgo: Double?

    enum CodingKeys: String, CodingKey {
        case symbol, name
        case earningsDate = "earnings_date"
        case earningsDateEnd = "earnings_date_end"
        case status
        case epsEstimate = "eps_estimate"
        case epsYearAgo = "eps_year_ago"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? "?"
        name = try c.decodeIfPresent(String.self, forKey: .name)
        earningsDate = try c.decodeIfPresent(String.self, forKey: .earningsDate) ?? ""
        earningsDateEnd = try c.decodeIfPresent(String.self, forKey: .earningsDateEnd)
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? "estimated"
        epsEstimate = try c.decodeIfPresent(Double.self, forKey: .epsEstimate)
        epsYearAgo = try c.decodeIfPresent(Double.self, forKey: .epsYearAgo)
    }

    var id: String { "\(symbol)|\(earningsDate)" }
}
