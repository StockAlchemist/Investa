import Foundation

struct User: Codable, Sendable, Identifiable, Equatable {
    let id: Int
    let username: String
    let alias: String?
    let isActive: Bool
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, username, alias
        case isActive = "is_active"
        case createdAt = "created_at"
    }

    var displayName: String {
        if let alias, !alias.isEmpty { return alias }
        return username
    }
}

/// Response from `POST /api/auth/login`.
struct Token: Codable, Sendable {
    let accessToken: String
    let tokenType: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case tokenType = "token_type"
    }
}
