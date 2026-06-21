import Foundation

/// `GET /api/projection` — forward portfolio-value projection (lognormal model).
struct Projection: Codable, Sendable {
    let available: Bool
    let currentValue: Double?
    /// Geometric (median) annualized return used for the projection, in percent.
    let annualReturnPct: Double?
    let annualVolatilityPct: Double?
    let currency: String?
    let horizons: [ProjectionHorizon]?

    enum CodingKeys: String, CodingKey {
        case available
        case currentValue = "current_value"
        case annualReturnPct = "annual_return_pct"
        case annualVolatilityPct = "annual_volatility_pct"
        case currency
        case horizons
    }
}

/// One horizon's projected value: the median plus 10/25/75/90th percentile bands.
struct ProjectionHorizon: Codable, Sendable, Identifiable {
    let years: Int
    let medianValue: Double
    let medianReturnPct: Double
    let expectedValue: Double
    let p10: Double
    let p25: Double
    let p75: Double
    let p90: Double

    var id: Int { years }

    enum CodingKeys: String, CodingKey {
        case years
        case medianValue = "median_value"
        case medianReturnPct = "median_return_pct"
        case expectedValue = "expected_value"
        case p10, p25, p75, p90
    }
}
