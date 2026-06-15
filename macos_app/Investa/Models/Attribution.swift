import Foundation

/// `GET /api/attribution` — gain/value contribution by sector and by stock.
struct Attribution: Codable, Sendable {
    let sectors: [SectorContribution]
    let stocks: [StockContribution]
    let totalGain: Double

    enum CodingKeys: String, CodingKey {
        case sectors, stocks
        case totalGain = "total_gain"
    }

    struct SectorContribution: Codable, Sendable, Identifiable {
        let sector: String
        let gain: Double
        let value: Double
        let contribution: Double
        var id: String { sector }
    }

    struct StockContribution: Codable, Sendable, Identifiable {
        let symbol: String
        let name: String
        let gain: Double
        let value: Double
        let sector: String
        let contribution: Double
        var id: String { symbol }
    }
}
