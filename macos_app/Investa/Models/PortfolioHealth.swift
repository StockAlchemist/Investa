import Foundation

/// `GET /api/portfolio_health` — overall score plus three component scores.
struct PortfolioHealth: Codable, Sendable {
    let overallScore: Double
    let rating: String
    let components: Components

    enum CodingKeys: String, CodingKey {
        case overallScore = "overall_score"
        case rating, components
    }

    struct Component: Codable, Sendable {
        let score: Double
        let label: String
    }

    struct Components: Codable, Sendable {
        let diversification: Component
        let efficiency: Component
        let stability: Component
    }
}
