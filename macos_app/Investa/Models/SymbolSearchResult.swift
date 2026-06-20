import Foundation

/// A result from `GET /api/search?q=` (symbol search for the command palette).
struct SymbolSearchResult: Decodable, Sendable, Identifiable {
    let symbol: String
    let name: String
    let type: String

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        symbol = raw["symbol"]?.stringValue ?? "?"
        name = raw["name"]?.stringValue ?? ""
        type = raw["type"]?.stringValue ?? ""
    }

    var id: String { symbol }
}
