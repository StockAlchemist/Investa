import Foundation

/// An item from `GET /api/watchlist`. Decoded tolerantly via `JSONValue` since
/// the backend mixes types and many analytics fields are null.
struct WatchlistItem: Decodable, Sendable, Identifiable {
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
    }

    private func d(_ k: String) -> Double? { raw[k]?.doubleValue }
    private func s(_ k: String) -> String? { raw[k]?.stringValue }

    var symbol: String { s("Symbol") ?? "?" }
    var note: String { s("Note") ?? "" }
    var name: String? { s("Name") }
    var currency: String? { s("Currency") }
    var price: Double? { d("Price") }
    var dayChange: Double? { d("Day Change") }
    var dayChangePct: Double? { d("Day Change %") }
    var marketCap: Double? { d("Market Cap") }
    var peRatio: Double? { d("PE Ratio") }
    /// Percent, e.g. 15.0 for a 15% yield. The route forwards `Dividend Rate`
    /// and `Trailing Dividend Yield` alongside the yield so the encoding can be
    /// settled against them — see `DividendYield`.
    var dividendYield: Double? {
        DividendYield.normalize(
            rawYield: d("Dividend Yield"),
            dividendRate: d("Dividend Rate"),
            price: price,
            trailingYield: d("Trailing Dividend Yield")
        )
    }
    var aiScore: Double? { d("ai_score") }
    var intrinsicValue: Double? { d("intrinsic_value") }
    var marginOfSafety: Double? { d("margin_of_safety") }
    var sentiment: Double? { d("ai_sentiment") }
    var catalystCount: Int { raw["ai_catalysts"]?.arrayValue?.count ?? 0 }
    var sparkline: [Double] { raw["Sparkline"]?.arrayValue?.compactMap { $0.doubleValue } ?? [] }

    var id: String { symbol }
}
