import SwiftUI

// MARK: - Actionable Insights computation (mirrors web dashboard/DashboardInsights.tsx)

enum InsightKind: String, Identifiable { case ripening, drift, undervalued; var id: String { rawValue } }
enum InsightTone { case pos, warn, alert, neutral }

struct RipeningLot: Identifiable {
    let id = UUID()
    let symbol: String
    let account: String?
    let date: String
    let daysRemaining: Int
    let quantity: Double
    let gain: Double
}

struct DriftBucket: Identifiable {
    let id = UUID()
    let dim: String
    let bucket: String
    let currentPct: Double
    let targetPct: Double
    let drift: Double
}

struct UndervaluedHolding: Identifiable {
    var id: String { symbol }
    let symbol: String
    var account: String?
    let mos: Double
    let intrinsic: Double?
    var marketValue: Double?
}

struct InsightSummary: Identifiable {
    var id: String { kind.rawValue }
    let kind: InsightKind
    let tone: InsightTone
    let icon: String
    let title: String
    let sub: String?
}

struct InsightDetails {
    var ripening: [RipeningLot] = []
    var drift: [DriftBucket] = []
    var undervalued: [UndervaluedHolding] = []
}

/// Sheet scope: the whole card opens `.all`; an individual row opens `.kind`.
enum InsightScope: Identifiable {
    case all
    case kind(InsightKind)
    var id: String { switch self { case .all: return "all"; case .kind(let k): return k.rawValue } }
}

private let insightDateFmt: DateFormatter = {
    let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
}()

private func isUnknownCategory(_ v: String?) -> Bool {
    guard let v else { return true }
    let s = v.trimmingCharacters(in: .whitespaces).uppercased()
    return s.isEmpty || s == "-" || s == "NONE" || s == "NULL" || s == "UNKNOWN"
        || s.hasPrefix("N/A") || s.hasPrefix("UNKNOWN")
}

/// Compute the insight summaries + underlying detail records (matches the web).
func computeInsights(holdings: [Holding], currency: String,
                     targets: [String: [String: Double]]) -> (summaries: [InsightSummary], details: InsightDetails) {
    let oneYear = 365.0, window = 30.0, driftAlert = 10.0, mosSignificant = 10.0
    let now = Date()
    var det = InsightDetails()

    // 1) Lots ripening to long-term within 30 days, with a positive gain.
    for h in holdings {
        for lot in h.raw["lots"]?.arrayValue ?? [] {
            guard let dStr = lot["Date"]?.stringValue,
                  let d = insightDateFmt.date(from: String(dStr.prefix(10))) else { continue }
            let heldDays = now.timeIntervalSince(d) / 86_400
            let remaining = oneYear - heldDays
            let gain = lot["Unreal. Gain"]?.doubleValue ?? 0
            if remaining > 0, remaining <= window, gain > 0 {
                det.ripening.append(RipeningLot(symbol: h.symbol, account: h.account, date: dStr,
                                                daysRemaining: Int(ceil(remaining)),
                                                quantity: lot["Quantity"]?.doubleValue ?? 0, gain: gain))
            }
        }
    }
    det.ripening.sort { $0.daysRemaining < $1.daysRemaining }

    // 2) Drift breaches: any bucket >= 10% off its target.
    if !targets.isEmpty {
        let dims: [(key: String, label: String)] = [
            ("quoteType", "Asset Type"), ("sector", "Sector"), ("country", "Country"),
        ]
        for dim in dims {
            guard let t = targets[dim.key], !t.isEmpty else { continue }
            var agg: [String: Double] = [:]
            for h in holdings {
                let v = max(0, h.marketValue(currency: currency) ?? 0)
                let raw: String?
                switch dim.key {
                case "country": raw = h.string("geography") ?? h.string("Country")
                case "sector": raw = h.string("Sector")
                default: raw = h.string("quoteType")
                }
                let cat = isUnknownCategory(raw) ? "Unknown" : raw!
                agg[cat, default: 0] += v
            }
            let tot = agg.values.reduce(0, +)
            if tot <= 0 { continue }
            for (bucket, targetPct) in t {
                let currentPct = (agg[bucket] ?? 0) / tot * 100
                let drift = currentPct - targetPct
                if abs(drift) >= driftAlert {
                    det.drift.append(DriftBucket(dim: dim.label, bucket: bucket,
                                                 currentPct: currentPct, targetPct: targetPct, drift: drift))
                }
            }
        }
        det.drift.sort { abs($0.drift) > abs($1.drift) }
    }

    // 3) Significantly undervalued holdings (margin of safety > 10%), one row per symbol.
    var bySymbol: [String: UndervaluedHolding] = [:]
    var order: [String] = []
    for h in holdings {
        guard let mos = h.double("margin_of_safety"), mos > mosSignificant else { continue }
        let mv = h.marketValue(currency: currency) ?? 0
        if var existing = bySymbol[h.symbol] {
            existing.marketValue = (existing.marketValue ?? 0) + mv
            existing.account = nil
            bySymbol[h.symbol] = existing
        } else {
            bySymbol[h.symbol] = UndervaluedHolding(symbol: h.symbol, account: h.account, mos: mos,
                                                    intrinsic: h.double("intrinsic_value"),
                                                    marketValue: mv > 0 ? mv : nil)
            order.append(h.symbol)
        }
    }
    det.undervalued = order.compactMap { bySymbol[$0] }.sorted { $0.mos > $1.mos }

    // Summaries (titles/subs match the web).
    var out: [InsightSummary] = []
    if !det.ripening.isEmpty {
        let n = det.ripening.count
        out.append(InsightSummary(kind: .ripening, tone: .warn, icon: "hourglass",
                                  title: "\(n) \(n == 1 ? "lot ripens" : "lots ripen") to long-term in the next 30 days",
                                  sub: "Holding past the 1-year mark unlocks the LTCG rate."))
    }
    if !det.drift.isEmpty {
        let worst = det.drift[0]; let more = det.drift.count - 1
        out.append(InsightSummary(kind: .drift, tone: .alert, icon: "exclamationmark.triangle",
                                  title: "\(worst.bucket) (\(worst.dim)) \(worst.drift > 0 ? "+" : "")\(String(format: "%.1f", worst.drift))% off target",
                                  sub: more > 0
                                       ? "\(more) more bucket\(more == 1 ? "" : "s") also breached — tap for the full list."
                                       : "See the Portfolio tab to rebalance."))
    }
    if !det.undervalued.isEmpty {
        let n = det.undervalued.count
        out.append(InsightSummary(kind: .undervalued, tone: .pos, icon: "diamond",
                                  title: "\(n) \(n == 1 ? "holding trades" : "holdings trade") below fair value",
                                  sub: "Margin of safety > 10% on your latest screen."))
    }
    return (out, det)
}

extension InsightTone {
    var color: Color {
        switch self { case .pos: return .up; case .alert: return .down; case .warn: return .orange; case .neutral: return .primary }
    }
}

// MARK: - Shared modal chrome

/// Standard sheet header: tinted icon tile, title, subtitle, and a close button.
private struct SheetHeader: View {
    let icon: String
    let tint: Color
    let title: String
    let subtitle: String
    let dismiss: () -> Void
    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: icon).font(.title2).foregroundStyle(tint)
                .frame(width: 44, height: 44)
                .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.title2.bold())
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).font(.title2).foregroundStyle(.secondary)
        }
        .padding(20)
    }
}

// MARK: - Insights detail sheet

struct InsightsDetailSheet: View {
    @Environment(\.dismiss) private var dismiss
    let scope: InsightScope
    let details: InsightDetails
    let summaries: [InsightSummary]
    let currency: String

    private var isAll: Bool { if case .all = scope { return true }; return false }
    private func shows(_ k: InsightKind) -> Bool { if case .kind(let kk) = scope { return kk == k }; return true }

    var body: some View {
        VStack(spacing: 0) {
            SheetHeader(icon: "lightbulb.fill", tint: .orange, title: "Insight Details",
                        subtitle: isAll
                            ? "\(summaries.count) signal\(summaries.count == 1 ? "" : "s") flagged across your portfolio"
                            : "Underlying records behind this signal",
                        dismiss: { dismiss() })
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    if shows(.ripening), !details.ripening.isEmpty { ripeningSection }
                    if shows(.drift), !details.drift.isEmpty { driftSection }
                    if shows(.undervalued), !details.undervalued.isEmpty { undervaluedSection }
                    if !isAll, currentEmpty {
                        VStack(spacing: 8) {
                            Image(systemName: "sparkles").font(.title).foregroundStyle(.tertiary)
                            Text("No records to show.").foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity).padding(.vertical, 40)
                    }
                }
                .padding(20)
            }
        }
        #if os(macOS)
        .frame(width: 640, height: 600)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

    private var currentEmpty: Bool {
        switch scope {
        case .kind(.ripening): return details.ripening.isEmpty
        case .kind(.drift): return details.drift.isEmpty
        case .kind(.undervalued): return details.undervalued.isEmpty
        default: return false
        }
    }

    private func sectionHeader(_ icon: String, _ tone: InsightTone, _ title: String, _ desc: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon).font(.callout).foregroundStyle(tone.color)
                .frame(width: 32, height: 32).background(tone.color.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.callout.bold())
                Text(desc).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var ripeningSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("hourglass", .warn, "Lots Ripening to Long-Term",
                          "Selling these lots after the date below qualifies for the long-term capital gains rate.")
            VStack(spacing: 0) {
                HStack {
                    Text("Symbol").frame(maxWidth: .infinity, alignment: .leading)
                    Text("Acquired").frame(width: 90, alignment: .leading)
                    Text("Qty").frame(width: 60, alignment: .trailing)
                    Text("Unrealized").frame(width: 90, alignment: .trailing)
                    Text("Days Left").frame(width: 70, alignment: .trailing)
                }
                .font(.system(size: 10, weight: .bold)).textCase(.uppercase).foregroundStyle(.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6).background(.background.secondary)
                ForEach(details.ripening) { lot in
                    HStack {
                        HStack(spacing: 5) {
                            Text(lot.symbol).fontWeight(.semibold)
                            if let a = lot.account { Text(a).font(.caption2).foregroundStyle(.secondary) }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                        Text(displayDate(lot.date)).foregroundStyle(.secondary).frame(width: 90, alignment: .leading)
                        Text(Fmt.number(lot.quantity, fractionDigits: 0)).frame(width: 60, alignment: .trailing)
                        Text("+\(Fmt.currency(lot.gain, code: currency))").foregroundStyle(Color.up).fontWeight(.semibold)
                            .frame(width: 90, alignment: .trailing)
                        Text("\(lot.daysRemaining)d").fontWeight(.bold).foregroundStyle(.orange).frame(width: 70, alignment: .trailing)
                    }
                    .font(.caption).monospacedDigit().padding(.horizontal, 10).padding(.vertical, 7)
                    Divider()
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private var driftSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("exclamationmark.triangle", .alert, "Allocation Drift",
                          "Buckets that have drifted 10% or more from their target weight.")
            ForEach(details.drift) { d in
                let overweight = d.drift > 0
                let tone: Color = overweight ? .down : .orange
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(d.bucket).font(.callout.bold())
                            Text(d.dim).font(.system(size: 10, weight: .semibold)).textCase(.uppercase).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("\(overweight ? "+" : "")\(String(format: "%.1f", d.drift))%")
                            .font(.callout.weight(.black)).monospacedDigit().foregroundStyle(tone)
                    }
                    HStack {
                        Text("Current ").foregroundStyle(.secondary) + Text(String(format: "%.1f%%", d.currentPct)).fontWeight(.semibold)
                        Spacer()
                        Text("Target ").foregroundStyle(.secondary) + Text(String(format: "%.1f%%", d.targetPct)).fontWeight(.semibold)
                    }.font(.caption).monospacedDigit()
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(.quaternary)
                            Capsule().fill(tone.opacity(0.7)).frame(width: geo.size.width * min(1, abs(d.drift) * 3 / 100))
                        }
                    }.frame(height: 6)
                }
                .padding(12)
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
            }
        }
    }

    private var undervaluedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("diamond", .pos, "Trading Below Fair Value",
                          "Holdings with a margin of safety greater than 10% on the latest screen.")
            VStack(spacing: 0) {
                HStack {
                    Text("Symbol").frame(maxWidth: .infinity, alignment: .leading)
                    Text("Intrinsic").frame(width: 100, alignment: .trailing)
                    Text("Position").frame(width: 100, alignment: .trailing)
                    Text("Margin of Safety").frame(width: 120, alignment: .trailing)
                }
                .font(.system(size: 10, weight: .bold)).textCase(.uppercase).foregroundStyle(.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6).background(.background.secondary)
                ForEach(details.undervalued) { u in
                    HStack {
                        HStack(spacing: 5) {
                            Text(u.symbol).fontWeight(.semibold)
                            if let a = u.account { Text(a).font(.caption2).foregroundStyle(.secondary) }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                        Text(u.intrinsic.map { Fmt.currency($0, code: currency) } ?? "—").foregroundStyle(.secondary)
                            .frame(width: 100, alignment: .trailing)
                        Text(u.marketValue.map { Fmt.currency($0, code: currency) } ?? "—").foregroundStyle(.secondary)
                            .frame(width: 100, alignment: .trailing)
                        Text(String(format: "%.1f%%", u.mos)).fontWeight(.bold).foregroundStyle(Color.up)
                            .frame(width: 120, alignment: .trailing)
                    }
                    .font(.caption).monospacedDigit().padding(.horizontal, 10).padding(.vertical, 7)
                    Divider()
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private func displayDate(_ iso: String) -> String {
        guard let d = insightDateFmt.date(from: String(iso.prefix(10))) else { return iso }
        let f = DateFormatter(); f.dateStyle = .medium; return f.string(from: d)
    }
}

// MARK: - Confirmed dividends sheet

struct ConfirmedDividendsSheet: View {
    @Environment(\.dismiss) private var dismiss
    let events: [DividendEvent]
    let currency: String
    let onSelectSymbol: (String) -> Void

    private var confirmed: [DividendEvent] {
        events.filter { $0.status == "confirmed" }.sorted { $0.dividendDate < $1.dividendDate }
    }
    private var total: Double { confirmed.reduce(0) { $0 + $1.amount } }

    var body: some View {
        VStack(spacing: 0) {
            SheetHeader(icon: "checkmark.seal.fill", tint: .up, title: "Confirmed Dividends",
                        subtitle: "\(confirmed.count) payment\(confirmed.count == 1 ? "" : "s") · \(Fmt.currency(total, code: currency)) total",
                        dismiss: { dismiss() })
            Divider()
            ScrollView {
                if confirmed.isEmpty {
                    Text("No confirmed dividends.").foregroundStyle(.secondary).padding(.vertical, 48)
                } else {
                    VStack(spacing: 2) {
                        ForEach(Array(confirmed.enumerated()), id: \.offset) { _, ev in
                            Button { onSelectSymbol(ev.symbol); dismiss() } label: {
                                HStack {
                                    HStack(spacing: 5) {
                                        Text(ev.symbol).fontWeight(.bold)
                                        Image(systemName: "checkmark.seal.fill").font(.caption2).foregroundStyle(Color.up)
                                    }
                                    Spacer()
                                    Text(displayDate(ev.dividendDate)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                                    Text(Fmt.currency(ev.amount, code: currency)).fontWeight(.bold).foregroundStyle(Color.up)
                                        .monospacedDigit().frame(width: 100, alignment: .trailing)
                                }
                                .padding(.horizontal, 12).padding(.vertical, 8)
                                .contentShape(Rectangle())
                                .rowHover()
                            }.buttonStyle(.plain)
                        }
                    }.padding(.horizontal, 12).padding(.vertical, 8)
                }
            }
        }
        #if os(macOS)
        .frame(width: 480, height: 560)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

    private func displayDate(_ iso: String) -> String {
        let inF = DateFormatter(); inF.locale = Locale(identifier: "en_US_POSIX"); inF.dateFormat = "yyyy-MM-dd"
        guard let d = inF.date(from: String(iso.prefix(10))) else { return iso }
        let f = DateFormatter(); f.dateStyle = .medium; return f.string(from: d)
    }
}

// MARK: - Portfolio health analysis sheet

struct HealthAnalysisSheet: View {
    @Environment(\.dismiss) private var dismiss
    let health: PortfolioHealth

    var body: some View {
        VStack(spacing: 0) {
            SheetHeader(icon: "waveform.path.ecg", tint: Theme.brand, title: "Portfolio Health Analysis",
                        subtitle: "Scoring Methodology", dismiss: { dismiss() })
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Weighting + legend
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Overall Score").font(.callout.bold())
                        Text("The overall score is a weighted average of three key pillars:")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack(spacing: 10) {
                            weightCell("40%", "Diversification", .up)
                            weightCell("40%", "Efficiency", .up)
                            weightCell("20%", "Stability", Theme.brand)
                        }
                        Text("Score Legend").font(.system(size: 10, weight: .bold)).textCase(.uppercase)
                            .foregroundStyle(.secondary).tracking(1).padding(.top, 4)
                        HStack(spacing: 14) {
                            legendDot(.red, "0-39 Critical"); legendDot(.yellow, "40-59 Fair")
                            legendDot(.up, "60-79 Good"); legendDot(Theme.brand, "80-100 Excellent")
                        }.font(.system(size: 10))
                    }
                    .padding(16).frame(maxWidth: .infinity, alignment: .leading)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))

                    componentSection("chart.pie.fill", .up, "Diversification (HHI)", health.components.diversification,
                                     "Measured using the Herfindahl-Hirschman Index (HHI). Lower is better, indicating less concentration.",
                                     "Score 80-100: <0.12 (Excellent) • Score 60-79: 0.12-0.35 (Moderate) • Score <60: >0.35 (Concentrated)")
                    componentSection("waveform.path.ecg", .up, "Efficiency (Sharpe Cost)", health.components.efficiency,
                                     "Return generated per unit of risk. Higher is better.",
                                     "Score 80-100: >1.0 (Excellent) • Score 30-79: 0-1.0 (Fair) • Score <30: <0 (Poor)")
                    componentSection("checkmark.shield.fill", Theme.brand, "Stability (Volatility)", health.components.stability,
                                     "Annualized volatility. Lower suggests steadier growth.",
                                     "Score 80-100: 5-25% (Ideal) • Score 40-79: <5% or 25-35% • Score <40: >35% (High)")
                }
                .padding(20)
            }
        }
        #if os(macOS)
        .frame(width: 540, height: 620)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

    private func weightCell(_ pct: String, _ label: String, _ tint: Color) -> some View {
        VStack(spacing: 3) {
            Text(pct).font(.callout.bold()).foregroundStyle(tint)
            Text(label).font(.system(size: 10)).textCase(.uppercase).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 8)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
    }

    private func legendDot(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 5) { Circle().fill(color).frame(width: 7, height: 7); Text(text) }
    }

    private func componentSection(_ icon: String, _ tint: Color, _ title: String,
                                  _ comp: PortfolioHealth.Component, _ desc: String, _ range: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Image(systemName: icon).font(.caption).foregroundStyle(tint)
                Text(title).font(.callout.bold())
                Spacer()
                if let m = comp.metric {
                    Text(m).font(.caption.monospaced())
                        .padding(.horizontal, 8).padding(.vertical, 2)
                        .background((comp.score >= 60 ? Color.up : .yellow).opacity(0.12), in: Capsule())
                        .foregroundStyle(comp.score >= 60 ? Color.up : .yellow)
                }
            }
            (Text("Score: ").foregroundStyle(.secondary)
                + Text("\(Int(comp.score))").fontWeight(.bold) + Text("/100").foregroundStyle(.secondary))
                .font(.caption)
            Text(desc).font(.caption).foregroundStyle(.secondary)
            Text(range).font(.system(size: 10)).foregroundStyle(.secondary)
                .padding(8).frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

// MARK: - Risk metric explanation sheet

struct RiskExplanation { let title: String; let description: String; let interpretation: String; let formula: String? }

let riskExplanations: [String: RiskExplanation] = [
    "Sharpe Ratio": RiskExplanation(
        title: "Sharpe Ratio",
        description: "Measures the performance of an investment compared to a risk-free asset, after adjusting for its risk.",
        interpretation: "A higher Sharpe ratio indicates better risk-adjusted performance. Generally, a ratio > 1 is considered good. (< 1: Suboptimal, 1 - 2: Good, 2 - 3: Very Good, > 3: Excellent)",
        formula: "(Rp - Rf) / σp"),
    "Sortino Ratio": RiskExplanation(
        title: "Sortino Ratio",
        description: "A variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility by using the asset's standard deviation of negative portfolio returns.",
        interpretation: "Like the Sharpe ratio, a higher result is better. It gives a more realistic view of downside risk for investors who don't mind upside volatility. (< 1: Bad, 1 - 2: Adequate, > 2: Great, > 3: Excellent)",
        formula: "(Rp - Rf) / σd"),
    "Volatility": RiskExplanation(
        title: "Annualized Volatility",
        description: "A statistical measure of the dispersion of returns for a given security or market index. In this context, it represents the annualized standard deviation.",
        interpretation: "Higher volatility means the price can change dramatically in a short time period in either direction. Lower volatility indicates steadier price action. (< 10%: Low Risk, 10-20%: Moderate, 20-30%: High, > 30%: Speculative)",
        formula: "std_dev(returns) * √252"),
    "Max Drawdown": RiskExplanation(
        title: "Maximum Drawdown",
        description: "The maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.",
        interpretation: "It is an indicator of downside risk over a specified time period. A lower (closer to 0%) drawdown suggests better capital preservation capabilities. (0-10%: Excellent, 10-20%: Good, 20-30%: Fair, > 30%: Concerning)",
        formula: "(Trough Value - Peak Value) / Peak Value"),
    "Beta": RiskExplanation(
        title: "Portfolio Beta",
        description: "Measures the price sensitivity of an investment relative to the overall market (S&P 500).",
        interpretation: "Beta = 1: Moves with the market. Beta > 1: More volatile than the market (Aggressive). Beta < 1: Less volatile than the market (Defensive).",
        formula: "Cov(Rp, Rm) / Var(Rm)"),
    "Alpha": RiskExplanation(
        title: "Jensen's Alpha",
        description: "The excess return of an investment relative to the return of a benchmark index after adjusting for risk.",
        interpretation: "Positive alpha indicates the investment outperformed its benchmark after adjusting for risk (Beta). It represents the value added by the portfolio manager/strategy.",
        formula: "Rp - [Rf + Beta * (Rm - Rf)]"),
]

struct MetricExplanationSheet: View {
    @Environment(\.dismiss) private var dismiss
    let metricKey: String

    var body: some View {
        let e = riskExplanations[metricKey]
        VStack(spacing: 0) {
            SheetHeader(icon: "info.circle.fill", tint: Theme.brand, title: e?.title ?? metricKey,
                        subtitle: "Metric Explanation", dismiss: { dismiss() })
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let e {
                        block("What is it?", e.description)
                        block("Interpretation", e.interpretation)
                        if let formula = e.formula {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Approx. Formula").font(.system(size: 10, weight: .bold)).textCase(.uppercase)
                                    .foregroundStyle(.secondary).tracking(1)
                                Text(formula).font(.callout.monospaced()).foregroundStyle(Theme.brand)
                                    .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                                    .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                }
                .padding(20)
            }
        }
        #if os(macOS)
        .frame(width: 460, height: 480)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
    }

    private func block(_ heading: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(heading).font(.system(size: 10, weight: .bold)).textCase(.uppercase)
                .foregroundStyle(.secondary).tracking(1)
            Text(body).font(.callout).foregroundStyle(.primary.opacity(0.9))
        }
    }
}
