import Foundation

/// The market-trend indicator: one index's close against its moving average.
///
/// **Advisory only** — `advisoryOnly` is always true. No strategy acts on this
/// signal; gating a stock book with it was measured and rejected. It is market
/// context, and the UI must not phrase it as an instruction.
///
/// `state` is the **active** reading, fixed at the last completed month-end.
/// `provisionalState` is what the comparison would say if the month ended today
/// — a preview of the *next* reading. Merging the two would let a mid-month
/// price be mistaken for the current state.
struct TrendSignal: Decodable, Sendable {
    enum State: String, Decodable, Sendable {
        case up = "in"
        case down = "out"

        var label: String {
            switch self {
            case .up: return "Uptrend"
            case .down: return "Downtrend"
            }
        }
    }

    /// Always true — kept in the model so the flag cannot be dropped silently.
    let advisoryOnly: Bool
    let signalSymbol: String
    /// Display name for the symbol, e.g. "S&P 500". Named by the backend so every
    /// client labels a market identically; optional only to stay decodable
    /// against a server that predates the field.
    let signalName: String?
    /// Zone the payload's dates were reckoned in (always a market clock).
    let marketTimezone: String?
    let state: State
    let smaMonths: Int
    /// Month-end close that set the active signal.
    let decisionDate: String
    let decisionClose: Double
    let sma: Double
    /// Month the active signal governs, as `YYYY-MM`.
    let governsMonth: String
    let provisionalState: State
    let provisionalSMA: Double
    let latestClose: Double
    let latestDate: String
    /// Close at which the next month-end decision flips the signal.
    let flipClose: Double
    /// Signed distance of the latest close from `flipClose`, in percent.
    let distancePct: Double?
    let wouldFlip: Bool
    let nextDecisionDate: String
    let history: [Point]

    struct Point: Decodable, Sendable, Identifiable {
        let date: String
        let close: Double
        let sma: Double?

        var id: String { date }
    }

    private enum CodingKeys: String, CodingKey {
        case advisoryOnly = "advisory_only"
        case signalSymbol = "signal_symbol"
        case signalName = "signal_name"
        case marketTimezone = "market_timezone"
        case state
        case smaMonths = "sma_months"
        case decisionDate = "decision_date"
        case decisionClose = "decision_close"
        case sma
        case governsMonth = "governs_month"
        case provisionalState = "provisional_state"
        case provisionalSMA = "provisional_sma"
        case latestClose = "latest_close"
        case latestDate = "latest_date"
        case flipClose = "flip_close"
        case distancePct = "distance_pct"
        case wouldFlip = "would_flip"
        case nextDecisionDate = "next_decision_date"
        case history
    }
}

/// A strategy's measured record. Every field is optional because the set of
/// statistics differs by strategy, and because a missing figure must render as
/// a dash rather than a confident zero.
struct StrategyBacktest: Decodable, Sendable {
    let window: String?
    let cagr: Double?
    let volatility: Double?
    let maxDrawdown: Double?
    let sharpe: Double?
    let trainCAGR: Double?
    let testCAGR: Double?
    let longWindow: String?
    let longCAGR: Double?

    private enum CodingKeys: String, CodingKey {
        case window, cagr, volatility, sharpe
        case maxDrawdown = "max_drawdown"
        case trainCAGR = "train_cagr"
        case testCAGR = "test_cagr"
        case longWindow = "long_window"
        case longCAGR = "long_cagr"
    }
}

struct RankingSleeveSpec: Decodable, Sendable {
    let qualityWeight: Double
    let topN: Int
    let maxPerSector: Int?
    let sectorDigits: Int
    let minMarketCap: Double?
    let rebalance: String

    private enum CodingKeys: String, CodingKey {
        case qualityWeight = "quality_weight"
        case topN = "top_n"
        case maxPerSector = "max_per_sector"
        case sectorDigits = "sector_digits"
        case minMarketCap = "min_market_cap"
        case rebalance
    }
}

/// One entry in the strategy catalogue.
///
/// `risks` travels with the definition rather than living in a footnote: a CAGR
/// shown without the drawdown beside it is the number that gets people hurt.
///
/// There is no trend sleeve and no leverage field. Strategies hold individual
/// common stock only, and the absence of those members is what keeps the client
/// from rendering an option the backend cannot express.
struct StrategyDefinition: Decodable, Sendable, Identifiable {
    let id: String
    let name: String
    let summary: String
    let sleeves: [String: Double]
    let backtest: StrategyBacktest
    let risks: [String]
    let isDefault: Bool
    let ranking: RankingSleeveSpec

    private enum CodingKeys: String, CodingKey {
        case id, name, summary, sleeves, backtest, risks, ranking
        case isDefault = "is_default"
    }
}

struct StrategyCatalogue: Decodable, Sendable {
    let strategies: [StrategyDefinition]
    let `default`: String
}

/// One line of a sleeve's target book.
struct StrategyPosition: Decodable, Sendable, Identifiable {
    /// Every position is common stock. No fund, cash-proxy or levered cases
    /// exist — the backend cannot emit them, and declaring them here would make
    /// the omission look like an oversight rather than the constraint it is.
    enum Role: String, Decodable, Sendable {
        case stock

        var label: String { "Stock" }
    }

    let symbol: String
    let name: String?
    let role: Role
    let weight: Double
    let amount: Double
    let price: Double?
    let shares: Int?
    let cost: Double?
    let score: Double?
    let industry: String?
    let note: String?

    var id: String { symbol }

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        symbol = raw["symbol"]?.stringValue ?? "?"
        name = raw["name"]?.stringValue
        role = Role(rawValue: raw["role"]?.stringValue ?? "stock") ?? .stock
        weight = raw["weight"]?.doubleValue ?? 0
        amount = raw["amount"]?.doubleValue ?? 0
        price = raw["price"]?.doubleValue
        shares = raw["shares"]?.doubleValue.map { Int($0) }
        cost = raw["cost"]?.doubleValue
        score = raw["score"]?.doubleValue
        industry = raw["industry"]?.stringValue
        note = raw["note"]?.stringValue
    }
}

struct StrategySleeve: Decodable, Sendable, Identifiable {
    /// Where each position's `price` came from. Membership always comes from
    /// the ranking snapshot; prices are live quotes, falling back to the
    /// snapshot's stored close when a quote is unavailable.
    enum PriceSource: String, Decodable, Sendable {
        case live, snapshot, mixed
    }

    let key: String
    let label: String
    let weight: Double
    let amount: Double
    /// How many names the rule asks for, against how many the ranking supplied.
    let positionsRequested: Int?
    let positionsFilled: Int?
    /// Sum of the position amounts. Below `amount` when the book is short.
    let amountAllocated: Double?
    let positions: [StrategyPosition]
    let runID: Int?
    let rankedAt: String?
    let priceSource: PriceSource?

    var id: String { key }

    private enum CodingKeys: String, CodingKey {
        case key, label, weight, amount, positions
        case positionsRequested = "positions_requested"
        case positionsFilled = "positions_filled"
        case amountAllocated = "amount_allocated"
        case runID = "run_id"
        case rankedAt = "ranked_at"
        case priceSource = "price_source"
    }
}

/// `GET /api/strategies/{id}/allocation` — what the rule says to hold today.
struct StrategyAllocation: Decodable, Sendable {
    let strategyID: String
    let name: String
    let capital: Double
    let asOf: String
    /// Age of the ranking snapshot in whole days; nil if it cannot be dated.
    let rankingAgeDays: Int?
    /// True once the snapshot is old enough that the batch worker has probably
    /// stopped. The endpoints keep serving the last good run either way, so
    /// without this a dead worker is indistinguishable from a healthy one.
    let rankingIsStale: Bool?
    /// True when the ranking produced fewer names than the rule calls for, so
    /// some capital is deliberately left unallocated rather than the weights
    /// being silently widened away from the backtested rule. A matching
    /// `warnings` entry says how short and by how much.
    let isShort: Bool?
    let sleeves: [StrategySleeve]
    let warnings: [String]

    private enum CodingKeys: String, CodingKey {
        case name, capital, sleeves, warnings
        case strategyID = "strategy_id"
        case asOf = "as_of"
        case rankingAgeDays = "ranking_age_days"
        case rankingIsStale = "ranking_is_stale"
        case isShort = "is_short"
    }
}
