import Foundation

/// A realized capital-gain row from `GET /api/capital_gains`. Uses currency-
/// suffixed display columns, so we keep the raw map and resolve at runtime.
struct CapitalGain: Codable, Sendable, Identifiable {
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        raw = try container.decode([String: JSONValue].self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }

    func double(_ key: String) -> Double? { raw[key]?.doubleValue }
    func string(_ key: String) -> String? { raw[key]?.stringValue }

    var date: String { string("Date") ?? "" }
    var symbol: String { string("Symbol") ?? "?" }
    var account: String { string("Account") ?? "" }
    var type: String { string("Type") ?? "" }
    var quantity: Double { double("Quantity") ?? 0 }
    var proceedsDisplay: Double { double("Total Proceeds (Display)") ?? 0 }
    var costBasisDisplay: Double { double("Total Cost Basis (Display)") ?? 0 }
    var realizedGainDisplay: Double { double("Realized Gain (Display)") ?? 0 }

    var id: String { "\(date)|\(symbol)|\(account)|\(quantity)|\(realizedGainDisplay)" }
}
