import Foundation

/// `GET /api/risk_metrics`. Keys are human-readable with spaces.
struct RiskMetrics: Codable, Sendable {
    let maxDrawdown: Double?
    let volatilityAnn: Double?
    let sharpe: Double?
    let sortino: Double?

    enum CodingKeys: String, CodingKey {
        case maxDrawdown = "Max Drawdown"
        case volatilityAnn = "Volatility (Ann.)"
        case sharpe = "Sharpe Ratio"
        case sortino = "Sortino Ratio"
    }
}
