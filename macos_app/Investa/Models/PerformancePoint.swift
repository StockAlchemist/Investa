import Foundation

/// One point in the `GET /api/history` performance series. Beyond the typed
/// fields, the endpoint returns dynamic benchmark keys (one per benchmark
/// symbol), so the raw map is retained for overlay access.
struct PerformancePoint: Codable, Sendable, Identifiable {
    let date: String
    let value: Double
    let twr: Double?
    let drawdown: Double?
    let raw: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case date = "date"
        case value = "value"
        case twr = "twr"
        case drawdown = "drawdown"
    }

    init(from decoder: Decoder) throws {
        let map = try decoder.singleValueContainer().decode([String: JSONValue].self)
        raw = map
        date = map["date"]?.stringValue ?? ""
        value = map["value"]?.doubleValue ?? 0
        twr = map["twr"]?.doubleValue
        drawdown = map["drawdown"]?.doubleValue
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(raw)
    }

    func benchmark(_ key: String) -> Double? { raw[key]?.doubleValue }

    var id: String { date }

    /// Parsed `Date` for charting; backend dates are ISO `yyyy-MM-dd` (or full ISO).
    var parsedDate: Date? {
        PerformancePoint.dayFormatter.date(from: String(date.prefix(10)))
    }

    private static let dayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}

/// What the performance graph plots.
enum PerformanceView: String, CaseIterable, Identifiable {
    case value = "Value"
    case twr = "TWR"
    case drawdown = "Drawdown"
    var id: String { rawValue }
}
