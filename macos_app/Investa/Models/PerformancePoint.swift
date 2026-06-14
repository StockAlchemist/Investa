import Foundation

/// One point in the `GET /api/history` performance series. The endpoint also
/// returns dynamic benchmark keys, which we ignore for the MVP chart.
struct PerformancePoint: Codable, Sendable, Identifiable {
    let date: String
    let value: Double
    let twr: Double?
    let drawdown: Double?

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
