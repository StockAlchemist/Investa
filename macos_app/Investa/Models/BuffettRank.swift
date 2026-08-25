import Foundation

/// Which valuation model a company was scored under.
///
/// A company is not compared against the whole market but against others scored
/// the same way: a bank's leverage is normal for a bank and alarming for an
/// industrial, so the two are never ranked on the same scale.
enum BuffettModel: String, Decodable, Sendable, CaseIterable, Identifiable {
    case generic, bank, insurer, reit

    var id: String { rawValue }

    var label: String {
        switch self {
        case .generic: return "Operating business"
        case .bank: return "Bank"
        case .insurer: return "Insurer"
        case .reit: return "REIT"
        }
    }

    /// Short form for the compact badge in list rows.
    var shortLabel: String {
        switch self {
        case .generic: return "OP"
        case .bank: return "BANK"
        case .insurer: return "INS"
        case .reit: return "REIT"
        }
    }
}

/// A row from `GET /api/buffett-rank`.
///
/// Decoded tolerantly via `JSONValue` for the same reason as `ScreenerResult`:
/// values arrive from SQLite, where an integer column may surface as an Int and
/// a NULL as a missing key rather than an explicit null.
struct BuffettRankRow: Decodable, Sendable, Identifiable {
    let symbol: String
    let name: String?
    let model: BuffettModel
    let rank: Int?
    let compositeScore: Double?
    let qualityScore: Double?
    /// Nil only where no quote or no reported earnings were available; the
    /// score itself is computed for every model, ranked within it.
    let valueScore: Double?
    let confidence: Double?
    let coverage: Double?
    let returnsOnCapital: Double?
    let financialStrength: Double?
    let predictability: Double?
    let growth: Double?
    let capitalAllocation: Double?
    let price: Double?
    let marketCap: Double?
    /// The two scored value inputs. There is no DCF-derived field any more —
    /// the margin of safety was measured against thirteen years of
    /// point-in-time rankings, found to be noise, and removed.
    let earningsYield: Double?
    let fcfYield: Double?
    let periodCount: Int?
    let latestPeriod: String?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        symbol = raw["symbol"]?.stringValue ?? "?"
        name = raw["name"]?.stringValue
        model = BuffettModel(rawValue: raw["model"]?.stringValue ?? "generic") ?? .generic
        rank = raw["rank"]?.doubleValue.map { Int($0) }
        compositeScore = raw["composite_score"]?.doubleValue
        qualityScore = raw["quality_score"]?.doubleValue
        valueScore = raw["value_score"]?.doubleValue
        confidence = raw["confidence"]?.doubleValue
        coverage = raw["coverage"]?.doubleValue
        returnsOnCapital = raw["returns_on_capital"]?.doubleValue
        financialStrength = raw["financial_strength"]?.doubleValue
        predictability = raw["predictability"]?.doubleValue
        growth = raw["growth"]?.doubleValue
        capitalAllocation = raw["capital_allocation"]?.doubleValue
        price = raw["price"]?.doubleValue
        marketCap = raw["market_cap"]?.doubleValue
        earningsYield = raw["earnings_yield"]?.doubleValue
        fcfYield = raw["fcf_yield"]?.doubleValue
        periodCount = raw["period_count"]?.doubleValue.map { Int($0) }
        latestPeriod = raw["latest_period"]?.stringValue
    }

    var id: String { symbol }

    /// True when the score was cut for incomplete data. Worth surfacing next to
    /// the score itself: a demoted company looks identical to an honestly
    /// mediocre one otherwise.
    var isConfidenceReduced: Bool {
        guard let confidence else { return false }
        return confidence < 0.999
    }

    /// Whether the value score uses a free-cash-flow yield at all.
    ///
    /// Owner earnings are not derived for banks or insurers, so the pipeline
    /// never computes one for them and `score_value` renormalises over the
    /// metrics that are present — their value score is the earnings yield
    /// alone. Worth distinguishing in the UI: a dash reads as data that went
    /// missing, when this is a model that does not use the input.
    var scoresFcfYield: Bool { model != .bank && model != .insurer }

    /// The five quality pillars in weighted order, for a compact breakdown.
    var pillars: [(label: String, value: Double?)] {
        [
            ("Returns", returnsOnCapital),
            ("Strength", financialStrength),
            ("Predictable", predictability),
            ("Growth", growth),
            ("Capital", capitalAllocation),
        ]
    }
}

/// One page of `GET /api/buffett-rank`, with the number of rows matching the
/// active filters. The count is what lets the client tell "last page" apart
/// from "no matches" — a short page alone cannot distinguish them.
struct BuffettRankPage: Decodable, Sendable {
    let total: Int
    let rows: [BuffettRankRow]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        total = (try? container.decode(Int.self, forKey: .total)) ?? 0
        rows = (try? container.decode([BuffettRankRow].self, forKey: .rows)) ?? []
    }

    private enum CodingKeys: String, CodingKey { case total, rows }
}

/// One page of `GET /api/buffett-rank/exclusions`.
struct BuffettExclusionPage: Decodable, Sendable {
    let total: Int
    let rows: [BuffettExclusion]

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        total = (try? container.decode(Int.self, forKey: .total)) ?? 0
        rows = (try? container.decode([BuffettExclusion].self, forKey: .rows)) ?? []
    }

    private enum CodingKeys: String, CodingKey { case total, rows }
}

/// A company kept out of the ranking, with the reasons it failed.
struct BuffettExclusion: Decodable, Sendable, Identifiable {
    let symbol: String
    let name: String?
    let model: String?
    let reasons: String
    let periodCount: Int?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        symbol = raw["symbol"]?.stringValue ?? "?"
        name = raw["name"]?.stringValue
        model = raw["model"]?.stringValue
        reasons = raw["reasons"]?.stringValue ?? ""
        periodCount = raw["period_count"]?.doubleValue.map { Int($0) }
    }

    var id: String { symbol }

    var reasonList: [String] {
        reasons.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: " ")
        }
    }
}

/// Metadata for one completed ranking run (`GET /api/buffett-rank/latest`).
struct BuffettRankRun: Decodable, Sendable {
    let runId: Int
    let startedAt: String?
    let finishedAt: String?
    let universeSize: Int?
    let rankedCount: Int?
    let excludedCount: Int?

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        runId = raw["run_id"]?.doubleValue.map { Int($0) } ?? 0
        startedAt = raw["started_at"]?.stringValue
        finishedAt = raw["finished_at"]?.stringValue
        universeSize = raw["universe_size"]?.doubleValue.map { Int($0) }
        rankedCount = raw["ranked_count"]?.doubleValue.map { Int($0) }
        excludedCount = raw["excluded_count"]?.doubleValue.map { Int($0) }
    }
}
