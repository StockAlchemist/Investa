import Foundation

/// A symbol whose stored price history is known to be unreliable.
///
/// The archive has always known this and only ever said so in a terminal. A
/// chart that steps 30x in the middle because a reverse split was never applied
/// looks exactly like a chart of a stock that fell 97%, and nothing on screen
/// told the two apart.
///
/// Mirrors `DataQualityFlag` in `web_app/lib/api.ts` and the `/api/data_quality`
/// response in `server/routes/market.py` — same wire names, same severities.
struct DataQualityFlag: Decodable, Sendable, Equatable {
    /// How sure the archive is that something is wrong.
    enum Severity: String, Decodable, Sendable {
        /// A split is on record that the prices do not reflect. Definitely wrong.
        case high
        /// A jump nothing explains. Worth knowing, and not proof — plenty of
        /// thin stocks really do move like that.
        case medium

        /// An unknown severity from a newer backend reads as the milder one, so
        /// an old client understates rather than alarms.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Severity(rawValue: raw) ?? .medium
        }
    }

    let symbol: String
    let severity: Severity
    let findings: Int
    let kinds: [String]
    /// The date the problem sits at, as a wire `yyyy-MM-dd` string. Formatted
    /// for display with `MarketTime.formatted(_:)`, never shown raw.
    let occurredOn: String?
    let detail: String?

    enum CodingKeys: String, CodingKey {
        case symbol, severity, findings, kinds, detail
        case occurredOn = "occurred_on"
    }
}

struct DataQualityResponse: Decodable, Sendable {
    let symbols: [String: DataQualityFlag]
    let count: Int
    /// False when nobody has run the scan yet — which is a different thing from
    /// "nothing is wrong", and would otherwise be indistinguishable from it.
    let scanned: Bool
}
