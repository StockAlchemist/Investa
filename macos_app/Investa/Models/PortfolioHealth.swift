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
        /// Human-readable underlying metric (e.g. an HHI value). The backend may
        /// emit it as a number or a string, so decode either into a display string.
        let metric: String?

        enum CodingKeys: String, CodingKey { case score, label, metric }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            score = try c.decodeIfPresent(Double.self, forKey: .score) ?? 0
            label = try c.decodeIfPresent(String.self, forKey: .label) ?? ""
            if let s = try? c.decode(String.self, forKey: .metric) {
                metric = s
            } else if let d = try? c.decode(Double.self, forKey: .metric) {
                metric = String(format: "%.2f", d)
            } else {
                metric = nil
            }
        }
    }

    struct Components: Codable, Sendable {
        let diversification: Component
        let efficiency: Component
        let stability: Component
    }
}
