import Foundation
import SwiftUI

/// The valuation / earnings / profitability / market block the backend derives
/// per symbol (`key_metrics` on /fundamentals), and the scale each reading is
/// judged against.
///
/// Mirrors `web_app/lib/metrics.ts` — same wire names, same midpoints, same
/// clamps — which in turn mirrors what `_fundamental_metrics` computes in
/// server/routes/market.py. A P/E of 31 must mean the same thing, and take the
/// same colour, on every client and on the heatmap.
///
/// `mid` values are the S&P 500's own median for that metric, so roughly half
/// the index falls either side and the scale actually discriminates.
struct StockMetric: Identifiable, Sendable {
    enum Group: String, CaseIterable, Sendable {
        case valuation = "Valuation"
        case earnings = "Earnings & Sales"
        case profitability = "Profitability"
        case market = "Market"
    }

    enum Format: Sendable {
        /// A plain multiple or ratio: "24.00".
        case ratio
        /// Money per share: "$8.71".
        case dollar
        /// A large money figure: "$466.8B".
        case cap
        /// Days until the next report, or since the last: "12d" / "3d ago".
        case days
        /// A percentage: "+27.00%".
        case percent
    }

    let field: String
    /// Deliberately shorter than the heatmap's label. The map names a metric in
    /// a whole dropdown row; a column of rows has about twelve characters before
    /// the value gets pushed off, and a truncated label is worse than an
    /// abbreviated one.
    let label: String
    let group: Group
    let mid: Double
    let clamp: Double
    let format: Format
    /// Lower is better (P/E, leverage, days to a report).
    var inverted: Bool = false
    /// The wire value is a fraction and is shown as a percentage.
    var isFraction: Bool = false
    /// A magnitude with a hard floor and no bad direction (dividend yield,
    /// revenue). It carries no sign and takes no verdict colour.
    var sequential: Bool = false

    var id: String { field }

    /// The reading in display units — percentages scaled, everything else as
    /// filed.
    func scaled(_ value: Double) -> Double { isFraction ? value * 100 : value }

    func formatted(_ value: Double) -> String {
        let v = scaled(value)
        switch format {
        case .ratio:  return String(format: "%.2f", v)
        case .dollar: return String(format: "$%.2f", v)
        case .cap:    return Self.compactCap(v)
        case .days:   return v >= 0 ? String(format: "%.0fd", v) : String(format: "%.0fd ago", -v)
        case .percent:
            // A magnitude has no direction to report: "+3.50%" reads as a change
            // in the yield rather than as the yield itself.
            return sequential ? String(format: "%.2f%%", v)
                              : String(format: "%@%.2f%%", v >= 0 ? "+" : "", v)
        }
    }

    /// Signed distance from the neutral point, positive meaning "better".
    func deviation(_ value: Double) -> Double {
        let v = scaled(value)
        return inverted ? mid - v : v - mid
    }

    /// Ink for the reading. Green beats a typical S&P 500 company on this
    /// measure, red trails it.
    ///
    /// `deadZone` is the share of `clamp` around the neutral point that stays
    /// plain: thirty-odd rows of verdicts read as noise, so the unremarkable
    /// ones recede. That is a display choice — the verdict itself is the same
    /// one the heatmap paints.
    func tone(_ value: Double, deadZone: Double = 0.15) -> Color {
        guard !sequential else { return .primary }
        let d = deviation(value)
        if abs(d) < clamp * deadZone { return .primary }
        return d >= 0 ? .up : .down
    }

    static func compactCap(_ v: Double) -> String {
        if v >= 1e12 { return String(format: "$%.1fT", v / 1e12) }
        if v >= 1e9  { return String(format: "$%.1fB", v / 1e9) }
        if v >= 1e6  { return String(format: "$%.0fM", v / 1e6) }
        return String(format: "$%.0f", v)
    }

    /// "58.4M" — a share count, which is not money and must not wear a currency
    /// sign the way `compactCap` does.
    static func compactCount(_ v: Double) -> String {
        if v >= 1e9 { return String(format: "%.2fB", v / 1e9) }
        if v >= 1e6 { return String(format: "%.1fM", v / 1e6) }
        if v >= 1e3 { return String(format: "%.0fK", v / 1e3) }
        return String(format: "%.0f", v)
    }
}

extension StockMetric {
    /// Every metric the detail window shows, in reading order.
    ///
    /// Performance is deliberately absent: the Chart tab plots it over any
    /// horizon, which beats eleven more rows of numbers.
    static let panel: [StockMetric] = [
        // --- Valuation. Centred on a typical large-cap reading: centring a P/E
        // on zero would paint every profitable company red.
        .init(field: "pe_ratio",   label: "P/E",         group: .valuation, mid: 25, clamp: 15,  format: .ratio, inverted: true),
        .init(field: "forward_pe", label: "Forward P/E", group: .valuation, mid: 17, clamp: 10,  format: .ratio, inverted: true),
        .init(field: "peg_ratio",  label: "PEG",         group: .valuation, mid: 2,  clamp: 1.5, format: .ratio, inverted: true),
        .init(field: "ps_ratio",   label: "P/S",         group: .valuation, mid: 3,  clamp: 2.5, format: .ratio, inverted: true),
        .init(field: "pb_ratio",   label: "P/B",         group: .valuation, mid: 4,  clamp: 3,   format: .ratio, inverted: true),
        .init(field: "p_fcf",      label: "P/FCF",       group: .valuation, mid: 24, clamp: 15,  format: .ratio, inverted: true),
        .init(field: "ev_ebitda",  label: "EV/EBITDA",   group: .valuation, mid: 15, clamp: 10,  format: .ratio, inverted: true),
        .init(field: "ev_sales",   label: "EV/Sales",    group: .valuation, mid: 4,  clamp: 3,   format: .ratio, inverted: true),
        .init(field: "dividend_yield", label: "Div Yield", group: .valuation, mid: 0, clamp: 5,  format: .percent,
              isFraction: true, sequential: true),

        // --- Earnings & sales. Zero is meaningful here (a loss, or a decline),
        // so these stay zero-centred.
        .init(field: "eps_ttm",      label: "EPS (TTM)",      group: .earnings, mid: 0, clamp: 15, format: .dollar),
        .init(field: "eps_qoq",      label: "EPS Q/Q",        group: .earnings, mid: 0, clamp: 50, format: .percent, isFraction: true),
        .init(field: "eps_growth_3y", label: "EPS Growth 3Y", group: .earnings, mid: 0, clamp: 30, format: .percent, isFraction: true),
        .init(field: "eps_growth_5y", label: "EPS Growth 5Y", group: .earnings, mid: 0, clamp: 30, format: .percent, isFraction: true),
        .init(field: "eps_surprise", label: "EPS Surprise",   group: .earnings, mid: 0, clamp: 10, format: .percent, isFraction: true),
        .init(field: "sales_ttm",    label: "Sales (TTM)",    group: .earnings, mid: 0, clamp: 100e9, format: .cap, sequential: true),
        .init(field: "sales_qoq",    label: "Sales Q/Q",      group: .earnings, mid: 0, clamp: 30, format: .percent, isFraction: true),
        .init(field: "sales_growth_3y", label: "Sales Growth 3Y", group: .earnings, mid: 0, clamp: 25, format: .percent, isFraction: true),
        .init(field: "sales_growth_5y", label: "Sales Growth 5Y", group: .earnings, mid: 0, clamp: 25, format: .percent, isFraction: true),

        // --- Profitability & balance sheet.
        .init(field: "roa",  label: "ROA",  group: .profitability, mid: 0, clamp: 20, format: .percent, isFraction: true),
        .init(field: "roe",  label: "ROE",  group: .profitability, mid: 0, clamp: 50, format: .percent, isFraction: true),
        .init(field: "roic", label: "ROIC", group: .profitability, mid: 0, clamp: 30, format: .percent, isFraction: true),
        .init(field: "gross_margin",     label: "Gross Margin",  group: .profitability, mid: 0, clamp: 60, format: .percent, isFraction: true),
        .init(field: "operating_margin", label: "Oper. Margin",  group: .profitability, mid: 0, clamp: 30, format: .percent, isFraction: true),
        .init(field: "net_margin",       label: "Net Margin",    group: .profitability, mid: 0, clamp: 25, format: .percent, isFraction: true),
        // Liquidity: more cover is better, and 1.0 is the classic solvency line.
        .init(field: "quick_ratio",   label: "Quick Ratio",   group: .profitability, mid: 1,   clamp: 1, format: .ratio),
        .init(field: "current_ratio", label: "Current Ratio", group: .profitability, mid: 1.5, clamp: 1, format: .ratio),
        // Both arrive as percent points, as the filings express them.
        .init(field: "lt_debt_equity", label: "LT Debt/Eq", group: .profitability, mid: 50, clamp: 50, format: .ratio, inverted: true),
        .init(field: "debt_equity",    label: "Debt/Eq",    group: .profitability, mid: 80, clamp: 80, format: .ratio, inverted: true),

        // --- Market & sentiment.
        .init(field: "relative_volume", label: "Rel. Volume", group: .market, mid: 1, clamp: 1, format: .ratio),
        .init(field: "float_short",     label: "Float Short", group: .market, mid: 5, clamp: 5, format: .percent,
              inverted: true, isFraction: true),
        // Yahoo's consensus runs 1 (strong buy) to 5 (sell), so lower is better.
        .init(field: "analyst_recom",  label: "Analyst Consensus", group: .market, mid: 2.5, clamp: 1.5, format: .ratio, inverted: true),
        // Days until the next report; imminent earnings read green.
        .init(field: "earnings_days",  label: "Next Earnings", group: .market, mid: 45, clamp: 45, format: .days, inverted: true),
    ]

    static func inGroup(_ group: Group) -> [StockMetric] { panel.filter { $0.group == group } }
}
