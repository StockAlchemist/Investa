import Foundation

/// A single portfolio holding. Core fields are typed; the backend also emits
/// currency-suffixed columns (e.g. `"Market Value (USD)"`) that we resolve at
/// runtime from the active display currency via `currencyValue(_:currency:)`.
struct Holding: Codable, Sendable, Identifiable {
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

    var symbol: String { string("Symbol") ?? "?" }
    var account: String? { string("Account") }
    var quantity: Double? { double("Quantity") }
    var sector: String? { string("Sector") }
    var dayChangePct: Double? { double("Day Change %") }
    var unrealizedGainPct: Double? { double("Unreal. Gain %") }
    var totalReturnPct: Double? { double("Total Return %") }
    var irrPct: Double? { double("IRR (%)") }

    /// Resolve a currency-suffixed column such as `"Market Value (USD)"`.
    /// Falls back to the unsuffixed key if the suffixed one is absent.
    func currencyValue(_ base: String, currency: String) -> Double? {
        double("\(base) (\(currency))") ?? double(base)
    }

    func marketValue(currency: String) -> Double? {
        currencyValue("Market Value", currency: currency)
    }

    var id: String { "\(symbol)|\(account ?? "")" }
}
