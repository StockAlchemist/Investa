import SwiftUI
import Charts

/// Shared card chrome.
private struct Card<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.headline)
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}

// MARK: - Portfolio hero (mirrors Dashboard.tsx PortfolioHeroCard)

enum HeroPeriod: String, CaseIterable, Identifiable {
    case day = "1D", wtd = "WTD", mtd = "MTD", ytd = "YTD", y1 = "1Y"
    var id: String { rawValue }
}

struct PortfolioHeroCard: View {
    let metrics: Metrics?
    let currency: String
    let longHistory: [PerformancePoint]
    @State private var period: HeroPeriod = .day

    private var dayGL: Double? { metrics?.dayChangeDisplay }
    private var dayGLPct: Double? { metrics?.dayChangePercent }

    private var periodView: (series: [Double], pct: Double?, abs: Double?) {
        if period == .day { return ([], dayGLPct, dayGL) }
        let rows = longHistory.compactMap { p -> (Date, Double)? in
            guard let d = p.parsedDate else { return nil }; return (d, p.value)
        }
        guard !rows.isEmpty else { return ([], nil, nil) }
        let cut = Self.cutoff(period)
        let beforeIdx = rows.firstIndex { $0.0 >= cut }
        let sliced = rows.filter { $0.0 >= cut }
        let anchor = (beforeIdx ?? 0) > 0 ? rows[(beforeIdx ?? 0) - 1] : (sliced.first ?? rows[0])
        let tail = sliced.isEmpty ? Array(rows.suffix(1)) : sliced
        let series = [anchor.1] + tail.map { $0.1 }
        let start = anchor.1, end = tail.last?.1 ?? anchor.1
        guard start != 0 else { return (series, nil, nil) }
        let absV = end - start
        return (series, absV / start * 100, absV)
    }

    var body: some View {
        let positive = (dayGL ?? 0) >= 0
        return VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Label("Total Portfolio Value", systemImage: "wallet.pass")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 12) {
                        Text(Fmt.currency(metrics?.marketValue, code: currency))
                            .font(.system(size: 40, weight: .black)).minimumScaleFactor(0.5).lineLimit(1)
                        if let g = dayGL {
                            HStack(spacing: 4) {
                                Image(systemName: positive ? "arrow.up.right" : "arrow.down.right")
                                Text("\(g >= 0 ? "+" : "")\(Fmt.currency(g, code: currency))").fontWeight(.semibold)
                                if let p = dayGLPct {
                                    Text("\(p >= 0 ? "+" : "")\(String(format: "%.2f%%", p))")
                                        .font(.caption.bold())
                                        .padding(.horizontal, 8).padding(.vertical, 2)
                                        .background((positive ? Color.green : .red).opacity(0.12), in: Capsule())
                                }
                                Text("today").font(.caption).foregroundStyle(.secondary)
                            }
                            .foregroundStyle(positive ? .green : .red)
                        }
                    }
                }
                Spacer()
                HStack(spacing: 0) {
                    statPill("Total TWR", metrics?.cumulativeTWR, sub: nil)
                    if let a = metrics?.annualizedTWR { Divider().frame(height: 30); statPill("Ann. TWR", a, sub: "p.a.") }
                    if let irr = metrics?.portfolioMWR { Divider().frame(height: 30); statPill("IRR (MWR)", irr, sub: "p.a.") }
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Picker("", selection: $period) {
                        ForEach(HeroPeriod.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented).fixedSize()
                    Spacer()
                    if period != .day, let pct = periodView.pct, let absV = periodView.abs {
                        Text("\(pct >= 0 ? "+" : "")\(String(format: "%.2f%%", pct)) (\(Fmt.currency(absV, code: currency)))")
                            .font(.caption.weight(.bold)).foregroundStyle(Fmt.tint(for: pct))
                    }
                }
                if periodView.series.count > 1 {
                    let up = (periodView.pct ?? 0) >= 0
                    Chart(Array(periodView.series.enumerated()), id: \.offset) { i, v in
                        AreaMark(x: .value("i", i), y: .value("v", v))
                            .foregroundStyle((up ? Color.green : .red).opacity(0.18))
                        LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(up ? .green : .red)
                    }
                    .chartXAxis(.hidden).chartYAxis(.hidden).frame(height: 48)
                }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func statPill(_ label: String, _ value: Double?, sub: String?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(Fmt.percent(value)).font(.title3.bold()).foregroundStyle(Fmt.tint(for: value))
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
        }
        .padding(.horizontal, 16)
    }

    private static func cutoff(_ period: HeroPeriod) -> Date {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let startOfDay = cal.startOfDay(for: Date())
        switch period {
        case .day: return startOfDay
        case .wtd:
            let dow = (cal.component(.weekday, from: startOfDay) + 5) % 7
            return cal.date(byAdding: .day, value: -dow, to: startOfDay) ?? startOfDay
        case .mtd:
            return cal.date(from: cal.dateComponents([.year, .month], from: startOfDay)) ?? startOfDay
        case .ytd:
            return cal.date(from: cal.dateComponents([.year], from: startOfDay)) ?? startOfDay
        case .y1:
            return cal.date(byAdding: .year, value: -1, to: startOfDay) ?? startOfDay
        }
    }
}

// MARK: - Today strip (mirrors dashboard/TodayStrip.tsx)

struct TodayStripCard: View {
    let holdings: [Holding]
    let currency: String
    let portfolioDayPct: Double?
    let indices: [IndexQuote]
    let onSelectSymbol: (String) -> Void

    private struct Mover { let symbol: String; let pct: Double; let contribution: Double }

    private func isCash(_ s: String) -> Bool {
        let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (")
    }

    private var movers: (gainers: [Mover], losers: [Mover]) {
        var bySymbol: [String: (mv: Double, pct: Double)] = [:]
        for h in holdings where !isCash(h.symbol) {
            guard let pct = h.dayChangePct else { continue }
            let mv = h.marketValue(currency: currency) ?? 0
            if var cur = bySymbol[h.symbol] { cur.mv += mv; bySymbol[h.symbol] = cur }
            else { bySymbol[h.symbol] = (mv, pct) }
        }
        let rows = bySymbol.map { Mover(symbol: $0.key, pct: $0.value.pct, contribution: $0.value.mv * $0.value.pct / 100) }
        let sorted = rows.sorted { $0.contribution > $1.contribution }
        return (Array(sorted.filter { $0.contribution > 0 }.prefix(3)),
                Array(sorted.filter { $0.contribution < 0 }.suffix(3).reversed()))
    }

    var body: some View {
        Card(title: "Today") {
            let m = movers
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 24) { marketCol; moverCol("Top Gainers", m.gainers, true); moverCol("Top Losers", m.losers, false) }
                VStack(alignment: .leading, spacing: 16) { marketCol; moverCol("Top Gainers", m.gainers, true); moverCol("Top Losers", m.losers, false) }
            }
        }
    }

    private var marketCol: some View {
        let top = indices.sorted { abs($0.changesPercentage ?? 0) > abs($1.changesPercentage ?? 0) }.prefix(3)
        return VStack(alignment: .leading, spacing: 6) {
            Label("Market Today", systemImage: "globe").font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            if let you = portfolioDayPct {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("\(you >= 0 ? "+" : "")\(String(format: "%.2f%%", you))")
                        .font(.title3.bold()).foregroundStyle(Fmt.tint(for: you))
                    Text("you").font(.caption2).foregroundStyle(.secondary)
                }
            }
            ForEach(Array(top)) { idx in
                HStack {
                    Text(idx.name ?? "Index").lineLimit(1)
                    Spacer()
                    Text(Fmt.percent(idx.changesPercentage)).fontWeight(.bold)
                        .foregroundStyle(Fmt.tint(for: idx.change))
                }.font(.caption)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func moverCol(_ title: String, _ rows: [Mover], _ positive: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: positive ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis")
                .font(.caption2).textCase(.uppercase).foregroundStyle(positive ? .green : .red)
            if rows.isEmpty {
                Text("No movers.").font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(rows, id: \.symbol) { r in
                    Button { onSelectSymbol(r.symbol) } label: {
                        HStack {
                            Text(r.symbol).fontWeight(.bold)
                            Spacer()
                            Text("\(r.pct >= 0 ? "+" : "")\(String(format: "%.2f%%", r.pct))")
                                .foregroundStyle(positive ? .green : .red)
                        }.font(.caption)
                    }.buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Upcoming dividends (mirrors dashboard/DashboardEvents.tsx)

struct UpcomingEventsCard: View {
    let dividends: [DividendEvent]
    let currency: String
    var windowDays = 14

    private static let inFmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
    }()

    private func relativeDay(_ iso: String) -> String {
        guard let d = Self.inFmt.date(from: String(iso.prefix(10))) else { return "" }
        let cal = Calendar.current
        let days = cal.dateComponents([.day], from: cal.startOfDay(for: Date()), to: cal.startOfDay(for: d)).day ?? 0
        switch days {
        case 0: return "today"
        case 1: return "tomorrow"
        case ..<0: return "\(-days)d ago"
        case 1..<7: return "\(days)d"
        default: return String(iso.prefix(10))
        }
    }

    private var upcoming: [DividendEvent] {
        let now = Calendar.current.startOfDay(for: Date())
        let cutoff = Calendar.current.date(byAdding: .day, value: windowDays, to: now) ?? now
        return dividends.filter {
            guard let d = Self.inFmt.date(from: String($0.dividendDate.prefix(10))) else { return false }
            return d >= now && d <= cutoff
        }.sorted { $0.dividendDate < $1.dividendDate }.prefix(8).map { $0 }
    }

    var body: some View {
        Card(title: "Upcoming Dividends") {
            if upcoming.isEmpty {
                Text("No dividend events in the next \(windowDays) days.").foregroundStyle(.secondary)
            } else {
                ForEach(upcoming) { ev in
                    HStack {
                        Text(ev.symbol).fontWeight(.bold)
                        if ev.status == "estimated" {
                            Label("est.", systemImage: "clock").font(.caption2).foregroundStyle(.orange)
                        } else {
                            Image(systemName: "checkmark.seal.fill").font(.caption2).foregroundStyle(.green)
                        }
                        Spacer()
                        Text(relativeDay(ev.dividendDate)).font(.caption).textCase(.uppercase).foregroundStyle(.secondary)
                        Text(Fmt.currency(ev.amount, code: currency)).fontWeight(.bold).foregroundStyle(.green)
                            .frame(width: 80, alignment: .trailing)
                    }
                    .padding(.vertical, 3)
                    Divider()
                }
            }
        }
    }
}

// MARK: - Actionable insights (mirrors dashboard/DashboardInsights.tsx)

struct ActionableInsightsCard: View {
    let holdings: [Holding]
    let currency: String

    private struct Insight { let icon: String; let title: String; let detail: String }

    private var insights: [Insight] {
        var out: [Insight] = []
        // Undervalued: holdings with a positive margin of safety (below fair value).
        let undervalued = holdings.filter { ($0.double("margin_of_safety") ?? 0) > 0 }
        if !undervalued.isEmpty {
            let names = undervalued.prefix(4).map { $0.symbol }.joined(separator: ", ")
            out.append(Insight(icon: "tag",
                               title: "\(undervalued.count) holding\(undervalued.count == 1 ? "" : "s") trade below fair value",
                               detail: names))
        }
        // Ripening lots: approaching 1-year (long-term) within 30 days, in gain.
        var ripening = 0
        let cal = Calendar.current
        for h in holdings {
            for lot in h.raw["lots"]?.arrayValue ?? [] {
                guard let dStr = lot["Date"]?.stringValue,
                      let d = Self.fmt.date(from: String(dStr.prefix(10))),
                      let gain = lot["Unreal. Gain"]?.doubleValue, gain > 0 else { continue }
                let held = cal.dateComponents([.day], from: d, to: Date()).day ?? 0
                let remaining = 365 - held
                if remaining > 0 && remaining <= 30 { ripening += 1 }
            }
        }
        if ripening > 0 {
            out.append(Insight(icon: "calendar.badge.clock",
                               title: "\(ripening) lot\(ripening == 1 ? "" : "s") ripen to long-term within 30 days",
                               detail: "Holding past the date qualifies for the long-term capital gains rate."))
        }
        return out
    }

    var body: some View {
        Card(title: "Actionable Insights") {
            if insights.isEmpty {
                Text("No ripening lots or new value buys.").font(.callout).foregroundStyle(.secondary)
            } else {
                ForEach(Array(insights.enumerated()), id: \.offset) { _, ins in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: ins.icon).foregroundStyle(.tint).frame(width: 18)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(ins.title).font(.callout.weight(.medium))
                            if !ins.detail.isEmpty { Text(ins.detail).font(.caption).foregroundStyle(.secondary) }
                        }
                        Spacer()
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }

    private static let fmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
}

// MARK: - Portfolio composition donut (metric + dimension toggle)

enum CompositionMetric: String, CaseIterable, Identifiable {
    case value = "Value", dayChange = "Day", gain = "Gain"
    var id: String { rawValue }
}

struct PortfolioCompositionCard: View {
    let holdings: [Holding]
    let currency: String
    @State private var metric: CompositionMetric = .value
    @State private var dimension = "Sector"

    private var slices: [AllocationSlice] {
        var totals: [String: Double] = [:]
        for h in holdings {
            let bucket = h.string(dimension) ?? "Unknown"
            let v: Double
            switch metric {
            case .value: v = h.marketValue(currency: currency) ?? 0
            case .dayChange: v = abs(h.currencyValue("Day Change", currency: currency) ?? 0)
            case .gain: v = abs(h.currencyValue("Unreal. Gain", currency: currency) ?? 0)
            }
            guard v > 0 else { continue }
            totals[bucket, default: 0] += v
        }
        return totals.map { AllocationSlice(label: $0.key, value: $0.value) }.sorted { $0.value > $1.value }
    }

    var body: some View {
        Card(title: "Portfolio Composition") {
            HStack {
                Picker("Metric", selection: $metric) {
                    ForEach(CompositionMetric.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).fixedSize()
                Spacer()
                Picker("By", selection: $dimension) {
                    ForEach(["Sector", "quoteType", "Country", "Industry"], id: \.self) {
                        Text($0 == "quoteType" ? "Asset Type" : $0).tag($0)
                    }
                }.pickerStyle(.menu).fixedSize()
            }
            if slices.isEmpty {
                Text("No data.").foregroundStyle(.secondary)
            } else {
                Chart(slices) { s in
                    SectorMark(angle: .value("Value", s.value), innerRadius: .ratio(0.6), angularInset: 1.5)
                        .foregroundStyle(by: .value("Label", s.label)).cornerRadius(3)
                }
                .frame(height: 260)
            }
        }
    }
}

// MARK: - Portfolio health

struct PortfolioHealthCard: View {
    let health: PortfolioHealth
    var body: some View {
        Card(title: "Portfolio Health") {
            HStack {
                Text(health.rating).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.0f", health.overallScore))
                    .font(.title.bold()).foregroundStyle(color(health.overallScore))
            }
            ForEach([
                ("Diversification", health.components.diversification),
                ("Efficiency", health.components.efficiency),
                ("Stability", health.components.stability),
            ], id: \.0) { name, comp in
                HStack {
                    Text(name).frame(width: 130, alignment: .leading)
                    ProgressView(value: max(0, min(comp.score, 100)), total: 100).tint(color(comp.score))
                    Text(String(format: "%.0f", comp.score)).monospacedDigit().frame(width: 36, alignment: .trailing)
                }
            }
        }
    }
    private func color(_ s: Double) -> Color { s < 40 ? .red : (s < 70 ? .orange : .green) }
}

// MARK: - Risk metrics

struct RiskMetricsCard: View {
    let risk: RiskMetrics?
    var body: some View {
        Card(title: "Risk Metrics") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                cell("Max Drawdown", Fmt.percent(risk?.maxDrawdown), .red)
                cell("Volatility (Ann.)", Fmt.percent(risk?.volatilityAnn), .primary)
                cell("Sharpe Ratio", Fmt.number(risk?.sharpe), .primary)
                cell("Sortino Ratio", Fmt.number(risk?.sortino), .primary)
            }
        }
    }
    private func cell(_ t: String, _ v: String, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(t).font(.caption).foregroundStyle(.secondary)
            Text(v).font(.title3.weight(.semibold)).foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Attribution

struct SectorAttributionCard: View {
    let attribution: Attribution
    let currency: String
    var body: some View {
        Card(title: "Contribution by Sector") {
            ForEach(attribution.sectors.sorted { $0.contribution > $1.contribution }) { s in
                row(s.sector, gain: s.gain, contribution: s.contribution)
            }
        }
    }
    private func row(_ label: String, gain: Double, contribution: Double) -> some View {
        VStack(spacing: 4) {
            HStack {
                Text(label).lineLimit(1)
                Spacer()
                Text(Fmt.currency(gain, code: currency)).monospacedDigit().foregroundStyle(Fmt.tint(for: gain))
                Text(String(format: "%.1f%%", contribution)).monospacedDigit().foregroundStyle(.secondary)
                    .frame(width: 56, alignment: .trailing)
            }
            Divider()
        }
        .padding(.vertical, 2)
    }
}

struct TopContributorsCard: View {
    let attribution: Attribution
    let currency: String
    var body: some View {
        Card(title: "Top Contributors") {
            ForEach(attribution.stocks.sorted { $0.contribution > $1.contribution }.prefix(10)) { s in
                VStack(spacing: 4) {
                    HStack {
                        Text(s.symbol).fontWeight(.medium).frame(width: 64, alignment: .leading)
                        Text(s.name).foregroundStyle(.secondary).lineLimit(1)
                        Spacer()
                        Text(Fmt.currency(s.gain, code: currency)).monospacedDigit().foregroundStyle(Fmt.tint(for: s.gain))
                        Text(String(format: "%.1f%%", s.contribution)).monospacedDigit().foregroundStyle(.secondary)
                            .frame(width: 56, alignment: .trailing)
                    }
                    Divider()
                }
                .padding(.vertical, 2)
            }
        }
    }
}

// MARK: - Allocation donut (reusable)

struct AllocationSlice: Identifiable {
    let id = UUID()
    let label: String
    let value: Double
}

/// Computes market-value allocation slices for a holding key (e.g. "Sector").
func allocationSlices(_ holdings: [Holding], key: String, currency: String) -> [AllocationSlice] {
    var totals: [String: Double] = [:]
    for h in holdings {
        let bucket = h.string(key) ?? "Unknown"
        let mv = h.marketValue(currency: currency) ?? 0
        guard mv > 0 else { continue }
        totals[bucket, default: 0] += mv
    }
    return totals.map { AllocationSlice(label: $0.key, value: $0.value) }.sorted { $0.value > $1.value }
}

struct AllocationDonutCard: View {
    let title: String
    let slices: [AllocationSlice]
    var showLegend = true
    private var total: Double { slices.reduce(0) { $0 + $1.value } }

    var body: some View {
        Card(title: title) {
            if slices.isEmpty {
                Text("No data").foregroundStyle(.secondary)
            } else {
                Chart(slices) { slice in
                    SectorMark(angle: .value("Value", slice.value),
                               innerRadius: .ratio(0.6), angularInset: 1.5)
                        .foregroundStyle(by: .value("Label", slice.label))
                        .cornerRadius(3)
                }
                .frame(height: 220)
                if showLegend {
                    ForEach(slices.prefix(8)) { s in
                        HStack {
                            Text(s.label).lineLimit(1)
                            Spacer()
                            Text(total > 0 ? String(format: "%.1f%%", s.value / total * 100) : "—")
                                .monospacedDigit().foregroundStyle(.secondary)
                        }
                        .font(.caption)
                    }
                }
            }
        }
    }
}

// MARK: - Projected income

struct ProjectedIncomeCard: View {
    let income: [ProjectedIncome]
    var body: some View {
        Card(title: "Projected Dividend Income (12 mo)") {
            if income.isEmpty {
                Text("No projected income.").foregroundStyle(.secondary)
            } else {
                Chart(income) { item in
                    BarMark(x: .value("Month", item.month), y: .value("Income", item.value))
                        .foregroundStyle(.tint)
                }
                .frame(height: 200)
            }
        }
    }
}

// MARK: - Dividend calendar

struct DividendCalendarCard: View {
    let events: [DividendEvent]
    let currency: String
    var body: some View {
        Card(title: "Upcoming Dividends") {
            if events.isEmpty {
                Text("No upcoming dividends.").foregroundStyle(.secondary)
            } else {
                ForEach(events.sorted { $0.exDividendDate < $1.exDividendDate }.prefix(10)) { ev in
                    VStack(spacing: 4) {
                        HStack {
                            Text(ev.symbol).fontWeight(.medium).frame(width: 70, alignment: .leading)
                            VStack(alignment: .leading) {
                                Text("Ex: \(ev.exDividendDate)").font(.caption)
                                Text("Pay: \(ev.dividendDate)").font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if ev.status == "estimated" {
                                Text("est.").font(.caption2).foregroundStyle(.orange)
                            }
                            Text(Fmt.currency(ev.amount, code: currency)).monospacedDigit()
                        }
                        Divider()
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }
}
