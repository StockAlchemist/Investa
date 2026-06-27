import SwiftUI
import Charts

/// Shared section chrome for the Performance tab.
private struct PSection<Content: View>: View {
    let title: String
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text(title).font(.headline); Spacer(); if let trailing { trailing } }
            content
            Spacer(minLength: 0)
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}

/// Helpers over the `/asset_change` payload (period → rows of dynamic keys).
enum AssetChangeData {
    /// (date, value) series for `"<series> <P>-Return"` within a period.
    static func returns(_ data: [String: [[String: JSONValue]]], period: String, series: String) -> [(String, Double)] {
        (data[period] ?? []).compactMap { row in
            guard let d = row["Date"]?.stringValue,
                  let v = row["\(series) \(period)-Return"]?.doubleValue else { return nil }
            return (d, v)
        }
    }
    static func year(_ iso: String) -> Int? { Int(iso.prefix(4)) }
    static func monthIndex(_ iso: String) -> Int? { Int(iso.dropFirst(5).prefix(2)).map { $0 - 1 } }
}

// MARK: - KPI strip (mirrors performance/KpiStrip.tsx)

struct PerfKpiStrip: View {
    let data: [String: [[String: JSONValue]]]
    let metrics: Metrics?
    let risk: RiskMetrics?
    let benchmarks: [String]

    private struct M { var ytd: Double?; var oneYear: Double?; var winRate: Double?
        var best: (String, Double)?; var worst: (String, Double)?; var maxDD: Double?; var vsBench: Double? }

    private func compounded(_ xs: [Double]) -> Double { (xs.reduce(1.0) { $0 * (1 + $1 / 100) } - 1) * 100 }

    private var m: M {
        var out = M()
        let monthly = AssetChangeData.returns(data, period: "M", series: "Portfolio")
        out.ytd = metrics?.ytdReturn
        if out.ytd == nil, !monthly.isEmpty {
            let latestYr = monthly.compactMap { AssetChangeData.year($0.0) }.max() ?? Calendar.current.component(.year, from: Date())
            let ytdVals = monthly.filter { AssetChangeData.year($0.0) == latestYr }.map { $0.1 }
            out.ytd = compounded(ytdVals)
        }
        let last12 = Array(monthly.suffix(12))
        if !last12.isEmpty { out.oneYear = compounded(last12.map { $0.1 }) }
        if !monthly.isEmpty {
            out.winRate = Double(monthly.filter { $0.1 > 0 }.count) / Double(monthly.count) * 100
            out.best = monthly.max { $0.1 < $1.1 }
            out.worst = monthly.min { $0.1 < $1.1 }
        }
        if let dd = risk?.maxDrawdown { out.maxDD = dd * 100 } else { out.maxDD = metrics?.maxDrawdown }
        if let bench = benchmarks.first {
            let bm = Array(AssetChangeData.returns(data, period: "M", series: bench).suffix(12))
            if !last12.isEmpty, !bm.isEmpty { out.vsBench = compounded(last12.map { $0.1 }) - compounded(bm.map { $0.1 }) }
        }
        return out
    }

    private func fmtMonth(_ iso: String) -> String {
        guard let y = AssetChangeData.year(iso), let mi = AssetChangeData.monthIndex(iso) else { return iso }
        let names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        return "\(names[max(0, min(11, mi))]) '\(String(y).suffix(2))"
    }

    var body: some View {
        let mt = m
        PSection(title: "Performance KPIs") {
            KpiRow(count: benchmarks.first != nil ? 7 : 6, minTileWidth: 150) {
                tile("YTD", pct(mt.ytd), nil, Fmt.tint(for: mt.ytd))
                tile("1Y", pct(mt.oneYear), nil, Fmt.tint(for: mt.oneYear))
                tile("Win Rate", mt.winRate.map { String(format: "%.0f%%", $0) } ?? "–", "of months", (mt.winRate ?? 0) >= 50 ? .green : .orange)
                tile("Best Month", mt.best.map { pct($0.1) } ?? "–", mt.best.map { fmtMonth($0.0) }, .green)
                tile("Worst Month", mt.worst.map { pct($0.1) } ?? "–", mt.worst.map { fmtMonth($0.0) }, .red)
                tile("Max DD", mt.maxDD.map { String(format: "%.2f%%", $0) } ?? "–", nil, .orange)
                if let b = benchmarks.first {
                    tile("vs \(b)", pct(mt.vsBench), "last 12M", Fmt.tint(for: mt.vsBench))
                }
            }
        }
    }
    private func pct(_ v: Double?) -> String { v.map { "\($0 > 0 ? "+" : "")\(String(format: "%.2f%%", $0))" } ?? "–" }
    private func tile(_ label: String, _ value: String, _ sub: String?, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone).lineLimit(1)
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Returns chart (mirrors performance/ReturnsChart.tsx)

struct ReturnsChart: View {
    let data: [String: [[String: JSONValue]]]
    let currency: String

    @State private var period = "M"
    @State private var valueMode = false
    @State private var count = 12

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private var compact: Bool { false }
    #endif

    private let periods: [(key: String, label: String, def: Int)] = [
        ("Y", "Annual", 10), ("M", "Monthly", 12), ("W", "Weekly", 12), ("D", "Daily", 30),
    ]

    private struct Bar: Identifiable { let id = UUID(); let date: String; let series: String; let value: Double }

    private var rows: [[String: JSONValue]] { Array((data[period] ?? []).suffix(count)) }

    private var bars: [Bar] {
        let suffix = valueMode ? "\(period)-Value" : "\(period)-Return"
        var keys: [String] = []
        if let last = rows.last { keys = last.keys.filter { $0.hasSuffix(suffix) } }
        if valueMode { keys = keys.filter { $0.hasPrefix("Portfolio") } }
        keys.sort { a, b in a.hasPrefix("Portfolio") ? true : (b.hasPrefix("Portfolio") ? false : a < b) }
        var out: [Bar] = []
        for row in rows {
            guard let date = row["Date"]?.stringValue else { continue }
            let dateStr = String(date.prefix(10))
            let sDate = shortDate(dateStr, period: period)
            for k in keys {
                if let v = row[k]?.doubleValue {
                    out.append(Bar(date: sDate, series: k.replacingOccurrences(of: " \(suffix)", with: ""), value: v))
                }
            }
        }
        return out
    }
    
    private func shortDate(_ d: String, period: String) -> String {
        guard d.count >= 10 else { return d }
        let y = d.prefix(4)
        let m = d.dropFirst(5).prefix(2)
        let day = d.dropFirst(8).prefix(2)
        switch period {
        case "Y": return String(y)
        case "M":
            let mInt = Int(m) ?? 1
            let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            let mStr = months[max(0, min(11, mInt - 1))]
            return "\(mStr) '\(y.suffix(2))"
        case "W", "D":
            return "\(m)/\(day)"
        default: return String(d.prefix(10))
        }
    }

    private var distinctDates: [String] {
        var seen = Set<String>(); var out: [String] = []
        for b in bars where !seen.contains(b.date) { seen.insert(b.date); out.append(b.date) }
        return out
    }
    
    private var xAxisValues: [String] {
        let dates = distinctDates
        guard compact else { return dates }
        
        switch period {
        case "D":
            // 30 items -> show every 5th or 6th to prevent overlapping
            let step = max(1, dates.count / 5)
            return dates.enumerated().compactMap { $0.offset % step == 0 ? $0.element : nil }
        case "W":
            // 12 items -> show every 3rd
            let step = max(1, dates.count / 4)
            return dates.enumerated().compactMap { $0.offset % step == 0 ? $0.element : nil }
        case "Y":
            // 10 items -> show every 2nd or all if formatted short
            // We'll format short, but let's also stride if count is large
            let step = max(1, dates.count / 5)
            return dates.enumerated().compactMap { $0.offset % step == 0 ? $0.element : nil }
        default:
            return dates
        }
    }

    var body: some View {
        PSection(title: "Returns") {
            controls.padding(.bottom, 8)

            if bars.isEmpty {
                Text("No return data.").foregroundStyle(.secondary).frame(height: 240)
            } else {
                Chart(bars) { bar in
                    BarMark(
                        x: .value("Date", bar.date),
                        y: .value("Return", bar.value),
                        width: .ratio(0.95)
                    )
                    .foregroundStyle(by: .value("Series", bar.series))
                    .position(by: .value("Series", bar.series), axis: .horizontal, span: .ratio(0.9))
                }
                .chartXAxis {
                    AxisMarks(values: xAxisValues) { value in
                        if let dateStr = value.as(String.self) {
                            AxisValueLabel {
                                if compact && period == "M" {
                                    Text(String(dateStr.prefix(1)))
                                } else if compact && period == "Y" {
                                    Text("'" + String(dateStr.suffix(2)))
                                } else {
                                    Text(dateStr)
                                }
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks { v in
                        AxisGridLine()
                        AxisValueLabel {
                            if let d = v.as(Double.self) {
                                Text(formatAxis(d, isPercent: !valueMode))
                            }
                        }
                    }
                }
                .chartLegend(.visible)
                .chartHoverTooltip(distinctDates) { i in
                    let date = distinctDates[i]
                    let entries = bars.filter { $0.date == date }
                    guard !entries.isEmpty else { return nil }
                    return ChartTooltipContent(title: date, rows: entries.map {
                        ChartTooltipRow(label: $0.series,
                                        value: valueMode ? Fmt.number($0.value, fractionDigits: 0)
                                                         : String(format: "%.2f%%", $0.value))
                    })
                }
                .frame(height: 280)
            }
        }
    }

    private func formatAxis(_ v: Double, isPercent: Bool) -> String {
        let absD = abs(v)
        let divisor: Double
        let suffix: String
        if absD >= 1_000_000_000 {
            divisor = 1_000_000_000
            suffix = "B"
        } else if absD >= 1_000_000 {
            divisor = 1_000_000
            suffix = "M"
        } else if absD >= 1_000 {
            divisor = 1_000
            suffix = "K"
        } else {
            divisor = 1
            suffix = ""
        }
        
        if !isPercent {
            return "\(Fmt.number(v / divisor, fractionDigits: divisor > 1 ? 1 : 0))\(suffix)"
        } else {
            return "\(String(format: divisor > 1 ? "%.1f" : "%.0f", v / divisor))\(suffix)%"
        }
    }

    private var controls: some View {
        HStack(spacing: 6) {
            // Period toggle – segmented pills
            Picker("", selection: $period) {
                ForEach(periods, id: \.key) { p in
                    Text(periodLabel(p)).tag(p.key)
                }
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()

            Spacer(minLength: 0)

            // Period count stepper – custom compact [– N +]
            HStack(spacing: 0) {
                Button { count = max(1, count - 1) } label: {
                    Image(systemName: "minus")
                        .font(.caption2.weight(.medium))
                        .frame(width: 26, height: 26)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)

                Text("\(count)")
                    .font(.caption.weight(.semibold).monospacedDigit())
                    .frame(minWidth: 24)
                    .lineLimit(1)

                Button { count = min(200, count + 1) } label: {
                    Image(systemName: "plus")
                        .font(.caption2.weight(.medium))
                        .frame(width: 26, height: 26)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))

            // View mode toggle – %/currency
            Picker("", selection: $valueMode) { Text("%").tag(false); Text(currency).tag(true) }
                .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 90)
        }
        .onChange(of: period) { _, new in count = periods.first { $0.key == new }?.def ?? 12 }
    }

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSizeClass
    #endif

    /// Short label on compact iPhones, full label otherwise.
    private func periodLabel(_ p: (key: String, label: String, def: Int)) -> String {
        #if os(iOS)
        if hSizeClass == .compact { return p.key }
        #endif
        return p.label
    }
}

// MARK: - Monthly heatmap (mirrors performance/MonthlyHeatmap.tsx)

struct MonthlyHeatmap: View {
    let data: [String: [[String: JSONValue]]]
    private let months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif

    /// Full-width flexible cells when there's room (macOS, iPad); a scrollable
    /// fixed-width grid only on the narrow iPhone, where 12 columns can't fit.
    private var flexible: Bool {
        #if os(iOS)
        return hSize != .compact
        #else
        return true
        #endif
    }

    private var grid: (years: [Int], values: [Int: [Double?]], totals: [Int: Double]) {
        var byYear: [Int: [Double?]] = [:]
        for row in data["M"] ?? [] {
            guard let date = row["Date"]?.stringValue, let v = row["Portfolio M-Return"]?.doubleValue,
                  let y = AssetChangeData.year(date), let mi = AssetChangeData.monthIndex(date) else { continue }
            if byYear[y] == nil { byYear[y] = Array(repeating: nil, count: 12) }
            byYear[y]![mi] = v
        }
        let years = byYear.keys.sorted(by: >)
        var totals: [Int: Double] = [:]
        for y in years {
            let comp = byYear[y]!.compactMap { $0 }.reduce(1.0) { $0 * (1 + $1 / 100) }
            totals[y] = (comp - 1) * 100
        }
        return (years, byYear, totals)
    }

    private func cellColor(_ v: Double?) -> Color {
        guard let v else { return .gray.opacity(0.12) }
        if v >= 5 { return .green.opacity(0.8) }
        if v >= 1 { return .green.opacity(0.4) }
        if v > 0 { return .green.opacity(0.18) }
        if v == 0 { return .gray.opacity(0.2) }
        if v > -1 { return .red.opacity(0.18) }
        if v > -5 { return .red.opacity(0.4) }
        return .red.opacity(0.8)
    }

    var body: some View {
        let g = grid
        PSection(title: "Monthly Returns") {
            if g.years.isEmpty {
                Text("No monthly data.").foregroundStyle(.secondary)
            } else if flexible {
                // Span the full panel width: cells flex to share the available space.
                heatGrid(g, flexible: true)
            } else {
                // iPhone can't fit 12 columns at a readable size — keep it scrollable.
                ScrollView(.horizontal, showsIndicators: true) { heatGrid(g, flexible: false) }
            }
        }
    }

    @ViewBuilder private func heatGrid(_ g: (years: [Int], values: [Int: [Double?]], totals: [Int: Double]), flexible: Bool) -> some View {
        Grid(alignment: .center, horizontalSpacing: 3, verticalSpacing: 3) {
            GridRow {
                Text("Year").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
                ForEach(months, id: \.self) {
                    Text($0).font(.caption2).foregroundStyle(.secondary)
                        .frame(maxWidth: flexible ? .infinity : nil)
                }
                Text("Total").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
            ForEach(g.years, id: \.self) { y in
                GridRow {
                    Text(String(y)).font(.caption.weight(.bold))
                        .lineLimit(1).fixedSize()
                        .gridColumnAlignment(.leading)
                    ForEach(0..<12, id: \.self) { mi in
                        let v = g.values[y]?[mi] ?? nil
                        Text(v.map { String(format: "%.1f", $0) } ?? "")
                            .font(.caption2).monospacedDigit()
                            .frame(width: flexible ? nil : 40, height: 24)
                            .frame(maxWidth: flexible ? .infinity : nil, maxHeight: 24)
                            .background(cellColor(v), in: RoundedRectangle(cornerRadius: 4))
                    }
                    Text(g.totals[y].map { "\($0 > 0 ? "+" : "")\(String(format: "%.1f%%", $0))" } ?? "—")
                        .font(.caption.weight(.bold)).monospacedDigit()
                        .lineLimit(1).minimumScaleFactor(0.7)
                        .foregroundStyle(Fmt.tint(for: g.totals[y]))
                }
            }
            Divider().gridCellColumns(14)
            let avgs = monthlyAverages(g)
            GridRow {
                Text("Avg").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
                ForEach(0..<12, id: \.self) { mi in
                    let a = avgs[mi]
                    Text(a.map { String(format: "%.1f", $0) } ?? "")
                        .font(.caption2.weight(.bold)).monospacedDigit()
                        .frame(width: flexible ? nil : 40, height: 24)
                        .frame(maxWidth: flexible ? .infinity : nil, maxHeight: 24)
                        .background(cellColor(a), in: RoundedRectangle(cornerRadius: 4))
                }
                let overall = avgs.compactMap { $0 }
                Text(overall.isEmpty ? "—" : "\(overall.reduce(0, +) / Double(overall.count) > 0 ? "+" : "")\(String(format: "%.1f%%", overall.reduce(0, +) / Double(overall.count)))")
                    .font(.caption.weight(.bold)).monospacedDigit()
                    .lineLimit(1).minimumScaleFactor(0.7)
                    .foregroundStyle(Fmt.tint(for: overall.isEmpty ? nil : overall.reduce(0, +) / Double(overall.count)))
            }
        }
        .frame(maxWidth: flexible ? .infinity : nil)
    }

    /// Average return for each month index (0=Jan … 11=Dec) across all years; nil if no data.
    private func monthlyAverages(_ g: (years: [Int], values: [Int: [Double?]], totals: [Int: Double])) -> [Double?] {
        (0..<12).map { mi in
            let vals = g.years.compactMap { g.values[$0]?[mi] ?? nil }
            return vals.isEmpty ? nil : vals.reduce(0, +) / Double(vals.count)
        }
    }
}

// MARK: - Drawdown timeline (mirrors performance/DrawdownTimeline.tsx)

struct DrawdownTimeline: View {
    let history: [PerformancePoint]

    private var stats: (series: [(Date, Double)], maxDD: Double, trough: String?, longest: Int) {
        let rows = history.compactMap { p -> (Date, String, Double)? in
            guard let dd = p.drawdown, let d = p.parsedDate else { return nil }
            return (d, p.date, dd > 0 ? -dd : dd)
        }
        var worst = 0.0; var troughDate: String?
        var run = 0; var longest = 0
        for r in rows {
            if r.2 < worst { worst = r.2; troughDate = String(r.1.prefix(10)) }
            if r.2 < -0.01 { run += 1; longest = max(longest, run) } else { run = 0 }
        }
        return (rows.map { ($0.0, $0.2) }, worst, troughDate, longest)
    }

    var body: some View {
        let s = stats
        PSection(title: "Drawdown") {
            if s.series.isEmpty {
                Text("No drawdown history available.").foregroundStyle(.secondary).frame(height: 200)
            } else {
                HStack(spacing: 16) {
                    stat("Max Drawdown", String(format: "%.2f%%", s.maxDD), .red)
                    stat("Longest Underwater", "\(s.longest)d", .orange)
                }
                Chart(s.series, id: \.0) { item in
                    AreaMark(x: .value("Date", item.0), y: .value("Drawdown", item.1))
                        .foregroundStyle(.red.opacity(0.25))
                    LineMark(x: .value("Date", item.0), y: .value("Drawdown", item.1)).foregroundStyle(.red)
                }
                .chartHoverTooltip(s.series.map(\.0)) { i in
                    ChartTooltipContent(title: Self.ddFmt.string(from: s.series[i].0),
                                        rows: [ChartTooltipRow(color: .red, label: "Drawdown",
                                                               value: String(format: "%.2f%%", s.series[i].1))])
                }
                .frame(height: 200)
                if let t = s.trough { Text("Deepest trough on \(t)").font(.caption2).foregroundStyle(.secondary) }
            }
        }
    }
    private func stat(_ l: String, _ v: String, _ tone: Color) -> some View {
        VStack(alignment: .leading) {
            Text(l).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(v).font(.title3.bold()).foregroundStyle(tone)
        }
    }
    static let ddFmt: DateFormatter = { let f = DateFormatter(); f.dateStyle = .medium; return f }()
}

// MARK: - Benchmark scoreboard (mirrors performance/BenchmarkScoreboard.tsx)

/// One benchmark's active-management stats, computed server-side
/// (/benchmark_scoreboard) so web and native share one correctly-annualized source.
struct BenchmarkStat: Decodable, Identifiable {
    let name: String
    let alpha: Double
    let beta: Double
    let r2: Double
    let trackingError: Double
    let informationRatio: Double
    let excessReturn: Double
    var id: String { name }
    enum CodingKeys: String, CodingKey {
        case name, alpha, beta, r2
        case trackingError = "tracking_error"
        case informationRatio = "information_ratio"
        case excessReturn = "excess_return"
    }
}

struct BenchmarkScoreboardResponse: Decodable {
    let scoreboard: [BenchmarkStat]
}

struct BenchmarkScoreboard: View {
    @EnvironmentObject private var appState: AppState
    @State private var stats: [BenchmarkStat] = []
    @State private var period = "all"            // 1y / 3y / 5y / all
    @State private var accounts: Set<String> = [] // empty = all accounts
    @State private var loading = false
    private let api = APIClient.shared
    private let periodOptions = [("1Y", "1y"), ("3Y", "3y"), ("5Y", "5y"), ("All", "all")]

    var body: some View {
        PSection(title: "Vs Benchmark") {
            controls
            let data = stats
            if loading && data.isEmpty {
                ProgressView().controlSize(.small).frame(maxWidth: .infinity)
            } else if data.isEmpty {
                Text("Not enough history to compute risk-adjusted stats.").foregroundStyle(.secondary)
            } else {
                Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 6) {
                    GridRow {
                        Text("Benchmark").gridColumnAlignment(.leading).lineLimit(1).minimumScaleFactor(0.8)
                        Text("α").lineLimit(1)
                        Text("β").lineLimit(1)
                        Text("R²").lineLimit(1)
                        Text("TE").lineLimit(1)
                        Text("IR").lineLimit(1)
                        Text("Excess").lineLimit(1)
                    }.font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    Divider()
                    ForEach(data) { r in
                        GridRow {
                            Text(r.name).gridColumnAlignment(.leading).fontWeight(.medium).lineLimit(1).minimumScaleFactor(0.8)
                            Text(signed(r.alpha) + "%").foregroundStyle(Fmt.tint(for: r.alpha)).lineLimit(1).minimumScaleFactor(0.8)
                            Text(String(format: "%.2f", r.beta)).lineLimit(1).minimumScaleFactor(0.8)
                            Text(String(format: "%.2f", r.r2)).lineLimit(1).minimumScaleFactor(0.8)
                            Text(String(format: "%.1f%%", r.trackingError)).lineLimit(1).minimumScaleFactor(0.8)
                            Text(String(format: "%.2f", r.informationRatio)).lineLimit(1).minimumScaleFactor(0.8)
                            Text(signed(r.excessReturn) + "%").foregroundStyle(Fmt.tint(for: r.excessReturn)).lineLimit(1).minimumScaleFactor(0.8)
                        }.font(.caption).monospacedDigit()
                    }
                }
                Text("α annualized excess vs beta-adjusted benchmark · β sensitivity · R² correlation² · TE tracking error · IR excess ÷ TE.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .task(id: signature) { await load() }
    }

    // Panel-local period + account scope (independent of the global filters).
    private var controls: some View {
        HStack(spacing: 10) {
            Picker("", selection: $period) {
                ForEach(periodOptions, id: \.1) { Text($0.0).tag($0.1) }
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()
            Spacer()
            PopoverMenu(minWidth: 200) {
                MenuRow(title: "All Accounts") { accounts = [] }
                let groups = orderedGroups
                if !groups.isEmpty {
                    MenuSectionHeader("Groups")
                    ForEach(groups, id: \.name) { g in
                        MenuRow(title: g.name) { accounts = Set(g.accounts) }
                    }
                }
                MenuSectionHeader("Accounts")
                ForEach(individualAccounts, id: \.self) { acc in
                    MenuToggleRow(title: acc, isOn: accounts.contains(acc)) { toggle(acc) }
                }
            } label: {
                Text(accountsLabel).font(.caption).lineLimit(1)
            }
            .fixedSize()
        }
        .padding(.bottom, 2)
    }

    private var accountsLabel: String {
        if accounts.isEmpty { return "All Accounts" }
        return accounts.count == 1 ? accounts.first! : "\(accounts.count) accounts"
    }
    private func toggle(_ a: String) {
        if accounts.contains(a) { accounts.remove(a) } else { accounts.insert(a) }
    }
    private var individualAccounts: [String] { appState.allAccounts.filter { $0 != "All Accounts" } }
    private var orderedGroups: [(name: String, accounts: [String])] {
        let g = appState.accountGroups
        let order = appState.accountGroupOrder.isEmpty ? Array(g.keys).sorted() : appState.accountGroupOrder
        return order.compactMap { name in g[name].map { (name, $0) } }
    }

    private var signature: String {
        "\(appState.displayCurrency)|\(period)|\(accounts.sorted().joined(separator: ","))|\(appState.benchmarks.sorted().joined(separator: ","))"
    }

    private func load() async {
        loading = true
        defer { loading = false }
        let acctItems = APIClient.arrayQuery("accounts", accounts.isEmpty ? nil : Array(accounts))
        let query = [URLQueryItem(name: "currency", value: appState.displayCurrency),
                     URLQueryItem(name: "period", value: period)]
            + acctItems + APIClient.arrayQuery("benchmarks", appState.benchmarks)
        let resp: BenchmarkScoreboardResponse? = try? await api.get("/benchmark_scoreboard", query: query)
        stats = resp?.scoreboard ?? []
    }

    private func signed(_ v: Double) -> String { "\(v >= 0 ? "+" : "")\(String(format: "%.2f", v))" }
}
