import Foundation

/// `POST /api/portfolio/ai_review`. The backend returns a wide, loosely-typed
/// payload (plus rate-limit/error markers), so it's decoded tolerantly.
struct PortfolioAIReview: Decodable, Sendable {
    let summary: String?
    let recommendations: [String]?
    let generatedAt: String?
    let error: String?
    let warning: String?
    let message: String?
    let scorecard: Scorecard?
    let analysis: Analysis?
    let optimizations: [Optimization]

    struct Scorecard: Sendable {
        let businessQuality: Double?
        let valueDiscipline: Double?
        let thesisIntegrity: Double?
    }
    struct Analysis: Sendable {
        let businessQuality: String?
        let valueDiscipline: String?
        let thesisIntegrity: String?
        let actionableRecommendations: String?
    }
    struct Optimization: Sendable, Identifiable {
        let id = UUID()
        let type: String
        let title: String
        let description: String
        let symbol: String
        let action: String
        let priority: String
    }

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        summary = raw["summary"]?.stringValue
        recommendations = raw["recommendations"]?.arrayValue?.compactMap { $0.stringValue }
        generatedAt = raw["generated_at"]?.stringValue
        error = raw["error"]?.stringValue
        warning = raw["warning"]?.stringValue
        message = raw["message"]?.stringValue
        if let sc = raw["scorecard"]?.objectValue {
            scorecard = Scorecard(businessQuality: sc["business_quality"]?.doubleValue,
                                  valueDiscipline: sc["value_discipline"]?.doubleValue,
                                  thesisIntegrity: sc["thesis_integrity"]?.doubleValue)
        } else { scorecard = nil }
        if let a = raw["analysis"]?.objectValue {
            analysis = Analysis(businessQuality: a["business_quality"]?.stringValue,
                                valueDiscipline: a["value_discipline"]?.stringValue,
                                thesisIntegrity: a["thesis_integrity"]?.stringValue,
                                actionableRecommendations: a["actionable_recommendations"]?.stringValue)
        } else { analysis = nil }
        optimizations = (raw["optimizations"]?.arrayValue ?? []).compactMap { v in
            guard let o = v.objectValue else { return nil }
            return Optimization(type: o["type"]?.stringValue ?? "",
                                title: o["title"]?.stringValue ?? "",
                                description: o["description"]?.stringValue ?? "",
                                symbol: o["symbol"]?.stringValue ?? "",
                                action: o["action"]?.stringValue ?? "",
                                priority: o["priority"]?.stringValue ?? "")
        }
    }
}

/// One AI chat turn. `role` is "user" or "ai" to match the backend.
struct ChatMessage: Codable, Sendable, Identifiable, Equatable {
    enum Role: String, Codable, Sendable { case user, ai }
    let role: Role
    let text: String
    var id = UUID()

    enum CodingKeys: String, CodingKey { case role, text }
}
