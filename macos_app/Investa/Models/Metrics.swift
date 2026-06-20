import Foundation

/// Response wrapper for `GET /api/summary` and `GET /api/summary/headline`.
/// `metrics` is null when there's nothing to show (e.g. all accounts closed).
struct SummaryResponse: Codable, Sendable {
    let metrics: Metrics?
}

/// Portfolio-level metrics. The backend emits a wide, partly-dynamic dictionary
/// (currency-dependent and feature-flagged keys), so we keep the raw map and
/// expose typed accessors over it rather than a brittle fixed schema.
struct Metrics: Codable, Sendable {
    let raw: [String: JSONValue]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        raw = try container.decode([String: JSONValue].self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }

    func double(_ key: String) -> Double? { raw[key]?.doubleValue }
    func string(_ key: String) -> String? { raw[key]?.stringValue }
    func bool(_ key: String) -> Bool? { raw[key]?.boolValue }

    // Commonly used headline fields (see web_app/lib/api.ts PortfolioSummary).
    var marketValue: Double? { double("market_value") }
    var dayChangeDisplay: Double? { double("day_change_display") }
    var dayChangePercent: Double? { double("day_change_percent") }
    var unrealizedGain: Double? { double("unrealized_gain") }
    var realizedGain: Double? { double("realized_gain") }
    var totalGain: Double? { double("total_gain") }
    var totalReturnPct: Double? { double("total_return_pct") }
    var dividends: Double? { double("dividends") }
    var annualizedTWR: Double? { double("annualized_twr") }
    var cumulativeTWR: Double? { double("cumulative_twr") }
    var portfolioMWR: Double? { double("portfolio_mwr") }
    var cashBalance: Double? { double("cash_balance") }
    var maxDrawdown: Double? { double("max_drawdown") }
    var sharpeRatio: Double? { double("sharpe_ratio") }
    var allSelectedClosed: Bool { bool("all_selected_closed") ?? false }

    var commissions: Double? { double("commissions") }
    var taxes: Double? { double("taxes") }
    var fxGainLossDisplay: Double? { double("fx_gain_loss_display") }
    var fxGainLossPct: Double? { double("fx_gain_loss_pct") }
    var dividendReturnCumulative: Double? { double("dividend_return_cumulative") }
    var dividendReturnAnnualized: Double? { double("dividend_return_annualized") }
    var dividendYieldPct: Double? { double("dividend_yield_pct") }
    var ytdReturn: Double? { double("ytd_return") }
    var volatilityAnn: Double? { double("volatility_ann") }
    var beta: Double? { double("beta") }
    var estAnnualIncomeDisplay: Double? { double("est_annual_income_display") }
}
