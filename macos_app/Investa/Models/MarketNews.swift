import Foundation

/// `GET /api/markets/news`. Decoded tolerantly (fields may be missing/null).
struct MarketNewsItem: Decodable, Sendable, Identifiable {
    let title: String
    let summary: String
    let url: String
    let thumbnail: String?
    let provider: String
    let pubDate: String
    let symbol: String?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        title = raw["title"]?.stringValue ?? ""
        summary = raw["summary"]?.stringValue ?? ""
        url = raw["url"]?.stringValue ?? ""
        thumbnail = raw["thumbnail"]?.stringValue
        provider = raw["provider"]?.stringValue ?? ""
        pubDate = raw["pub_date"]?.stringValue ?? ""
        symbol = raw["symbol"]?.stringValue
    }

    /// Stable identity (url may repeat/empty, so fall back to a composite).
    var id: String { url.isEmpty ? "\(title)|\(provider)|\(pubDate)" : url }
}

/// One index quote from `GET /api/indices` (values keyed by index id).
/// Decoded tolerantly; includes a short sparkline series when present.
struct IndexQuote: Decodable, Sendable, Identifiable {
    let name: String?
    let price: Double?
    let change: Double?
    let changesPercentage: Double?
    let sparkline: [Double]

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        name = raw["name"]?.stringValue
        price = raw["price"]?.doubleValue
        change = raw["change"]?.doubleValue
        changesPercentage = raw["changesPercentage"]?.doubleValue
        sparkline = raw["sparkline"]?.arrayValue?.compactMap { $0.doubleValue } ?? []
    }

    var id: String { name ?? UUID().uuidString }
}
