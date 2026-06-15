import Foundation

/// `GET /api/risk_metrics`. Keys are human-readable with spaces.
struct RiskMetrics: Codable, Sendable {
    let maxDrawdown: Double?
    let volatilityAnn: Double?
    let sharpe: Double?
    let sortino: Double?
    let beta: Double?
    let alpha: Double?
    /// Year-to-date return as a fraction (e.g. 0.123 → 12.3%).
    let ytdReturn: Double?

    enum CodingKeys: String, CodingKey {
        case maxDrawdown = "Max Drawdown"
        case volatilityAnn = "Volatility (Ann.)"
        case sharpe = "Sharpe Ratio"
        case sortino = "Sortino Ratio"
        case beta = "Beta"
        case alpha = "Alpha"
        case ytdReturn = "YTD Return"
    }
}
