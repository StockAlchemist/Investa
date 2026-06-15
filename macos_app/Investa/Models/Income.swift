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
    let dividendDate: String
    let exDividendDate: String
    let amount: Double
    let status: String

    enum CodingKeys: String, CodingKey {
        case symbol
        case dividendDate = "dividend_date"
        case exDividendDate = "ex_dividend_date"
        case amount, status
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? "?"
        dividendDate = try c.decodeIfPresent(String.self, forKey: .dividendDate) ?? ""
        exDividendDate = try c.decodeIfPresent(String.self, forKey: .exDividendDate) ?? ""
        amount = try c.decodeIfPresent(Double.self, forKey: .amount) ?? 0
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? "estimated"
    }

    var id: String { "\(symbol)|\(exDividendDate)" }
}
