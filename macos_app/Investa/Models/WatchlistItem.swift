import Foundation

/// An item from `GET /api/watchlist`. Many analytics fields are optional/null.
struct WatchlistItem: Codable, Sendable, Identifiable {
    let symbol: String
    let note: String
    let name: String?
    let currency: String?
    let price: Double?
    let dayChange: Double?
    let dayChangePct: Double?

    enum CodingKeys: String, CodingKey {
        case symbol = "Symbol"
        case note = "Note"
        case name = "Name"
        case currency = "Currency"
        case price = "Price"
        case dayChange = "Day Change"
        case dayChangePct = "Day Change %"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? "?"
        note = try c.decodeIfPresent(String.self, forKey: .note) ?? ""
        name = try c.decodeIfPresent(String.self, forKey: .name)
        currency = try c.decodeIfPresent(String.self, forKey: .currency)
        price = try c.decodeIfPresent(Double.self, forKey: .price)
        dayChange = try c.decodeIfPresent(Double.self, forKey: .dayChange)
        dayChangePct = try c.decodeIfPresent(Double.self, forKey: .dayChangePct)
    }

    var id: String { symbol }
}
