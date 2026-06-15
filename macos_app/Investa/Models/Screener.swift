import Foundation

/// A row from `POST /api/screener/run` or `/screener/narrative`.
/// Decoded tolerantly via `JSONValue` because the backend mixes types
/// (e.g. `has_ai_review` arrives as an integer 0/1 from SQLite, numbers may
/// arrive as strings, etc.).
struct ScreenerResult: Decodable, Sendable, Identifiable {
    let symbol: String
    let name: String?
    let price: Double?
    let intrinsicValue: Double?
    let marginOfSafety: Double?
    let peRatio: Double?
    let marketCap: Double?
    let sector: String?
    let hasAIReview: Bool?
    let aiScore: Double?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        symbol = raw["symbol"]?.stringValue ?? "?"
        name = raw["name"]?.stringValue
        price = raw["price"]?.doubleValue
        intrinsicValue = raw["intrinsic_value"]?.doubleValue
        marginOfSafety = raw["margin_of_safety"]?.doubleValue
        peRatio = raw["pe_ratio"]?.doubleValue
        marketCap = raw["market_cap"]?.doubleValue
        sector = raw["sector"]?.stringValue
        hasAIReview = raw["has_ai_review"]?.boolValue
        aiScore = raw["ai_score"]?.doubleValue
    }

    var id: String { symbol }

    /// Non-optional accessor for table sorting (Optional isn't Comparable).
    var sortMarginOfSafety: Double { marginOfSafety ?? -.greatestFiniteMagnitude }
}

/// AI audit returned by `POST /api/screener/review/{symbol}`. Decoded tolerantly.
struct ScreenReview: Decodable, Sendable {
    let scorecard: [String: Double]?
    let summary: String?
    let analysis: [String: String]?
    let error: String?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        summary = raw["summary"]?.stringValue
        error = raw["error"]?.stringValue
        if let card = raw["scorecard"]?.objectValue {
            scorecard = card.compactMapValues { $0.doubleValue }
        } else { scorecard = nil }
        if let a = raw["analysis"]?.objectValue {
            analysis = a.compactMapValues { $0.stringValue }
        } else { analysis = nil }
    }
}

/// `GET /api/watchlists` — list of saved watchlists.
struct WatchlistMeta: Codable, Sendable, Identifiable {
    let id: Int
    let name: String
}

/// Body for `POST /api/screener/run`.
struct ScreenerRequest: Encodable, Sendable {
    let universe_type: String
    let universe_id: String?
    let manual_symbols: [String]
    let fast_mode: Bool
}

enum ScreenerUniverse: String, CaseIterable, Identifiable {
    case watchlist, narrative, holdings, sp500, sp400, russell2000, all, manual
    var id: String { rawValue }
    var label: String {
        switch self {
        case .watchlist: return "Watchlist"
        case .narrative: return "Narrative Search (AI) ✨"
        case .holdings: return "Holdings"
        case .sp500: return "S&P 500 (Large Cap)"
        case .sp400: return "S&P 400 (Mid Cap)"
        case .russell2000: return "Russell 2000 (Small Cap)"
        case .all: return "All Database Stocks"
        case .manual: return "Custom List"
        }
    }
}
