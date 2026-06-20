import Foundation

/// A dividend record from `GET /api/dividends`.
struct Dividend: Codable, Sendable, Identifiable {
    let date: String
    let symbol: String
    let account: String
    let localCurrency: String
    let amountLocal: Double
    let fxRate: Double?
    let amountDisplay: Double
    let taxLocal: Double?
    let taxDisplay: Double?

    enum CodingKeys: String, CodingKey {
        case date = "Date"
        case symbol = "Symbol"
        case account = "Account"
        case localCurrency = "LocalCurrency"
        case amountLocal = "DividendAmountLocal"
        case fxRate = "FXRateUsed"
        case amountDisplay = "DividendAmountDisplayCurrency"
        case taxLocal = "TaxAmountLocal"
        case taxDisplay = "TaxAmountDisplayCurrency"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        date = try c.decodeIfPresent(String.self, forKey: .date) ?? ""
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? ""
        account = try c.decodeIfPresent(String.self, forKey: .account) ?? ""
        localCurrency = try c.decodeIfPresent(String.self, forKey: .localCurrency) ?? "USD"
        amountLocal = try c.decodeIfPresent(Double.self, forKey: .amountLocal) ?? 0
        fxRate = try c.decodeIfPresent(Double.self, forKey: .fxRate)
        amountDisplay = try c.decodeIfPresent(Double.self, forKey: .amountDisplay) ?? 0
        taxLocal = try c.decodeIfPresent(Double.self, forKey: .taxLocal)
        taxDisplay = try c.decodeIfPresent(Double.self, forKey: .taxDisplay)
    }

    var id: String { "\(date)|\(symbol)|\(account)|\(amountLocal)" }
}
