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
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 12)], spacing: 12) {
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
            for k in keys {
                if let v = row[k]?.doubleValue {
                    out.append(Bar(date: String(date.prefix(10)), series: k.replacingOccurrences(of: " \(suffix)", with: ""), value: v))
                }
            }
        }
        return out
    }

    private var distinctDates: [String] {
        var seen = Set<String>(); var out: [String] = []
        for b in bars where !seen.contains(b.date) { seen.insert(b.date); out.append(b.date) }
        return out
    }

    var body: some View {
        PSection(title: "Returns") {
            ScrollView(.horizontal, showsIndicators: false) {
                controls.padding(.bottom, 8)
            }
            if bars.isEmpty {
                Text("No return data.").foregroundStyle(.secondary).frame(height: 240)
            } else {
                Chart(bars) { bar in
                    BarMark(x: .value("Date", bar.date), y: .value("Return", bar.value))
                        .foregroundStyle(by: .value("Series", bar.series))
                        .position(by: .value("Series", bar.series))
                }
                .chartYAxis {
                    AxisMarks { v in
                        AxisGridLine()
                        AxisValueLabel {
                            if let d = v.as(Double.self) {
                                Text(valueMode ? Fmt.number(d, fractionDigits: 0) : String(format: "%.0f%%", d))
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

    private var controls: some View {
        HStack(spacing: 8) {
            Picker("", selection: $period) { ForEach(periods, id: \.key) { Text($0.label).tag($0.key) } }
                .pickerStyle(.menu).fixedSize()
                .onChange(of: period) { _, new in count = periods.first { $0.key == new }?.def ?? 12 }
            Stepper("Show \(count)", value: $count, in: 1...200).fixedSize()
            Picker("", selection: $valueMode) { Text("%").tag(false); Text(currency).tag(true) }
                .pickerStyle(.segmented).fixedSize()
        }
    }
}

// MARK: - Monthly heatmap (mirrors performance/MonthlyHeatmap.tsx)

struct MonthlyHeatmap: View {
    let data: [String: [[String: JSONValue]]]
    private let months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    Grid(alignment: .center, horizontalSpacing: 3, verticalSpacing: 3) {
                        GridRow {
                            Text("Year").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
                            ForEach(months, id: \.self) { Text($0).font(.caption2).foregroundStyle(.secondary) }
                            Text("Total").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                        }
                        ForEach(g.years, id: \.self) { y in
                            GridRow {
                                Text(String(y)).font(.caption.weight(.bold)).gridColumnAlignment(.leading)
                                ForEach(0..<12, id: \.self) { mi in
                                    let v = g.values[y]?[mi] ?? nil
                                    Text(v.map { String(format: "%.1f", $0) } ?? "")
                                        .font(.caption2).monospacedDigit()
                                        .frame(width: 40, height: 24)
                                        .background(cellColor(v), in: RoundedRectangle(cornerRadius: 4))
                                }
                                Text(g.totals[y].map { "\($0 > 0 ? "+" : "")\(String(format: "%.1f%%", $0))" } ?? "—")
                                    .font(.caption.weight(.bold)).monospacedDigit()
                                    .foregroundStyle(Fmt.tint(for: g.totals[y]))
                            }
                        }
                    }
                }
            }
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
            Menu {
                Button("All Accounts") { accounts = [] }
                if !appState.allAccounts.isEmpty { Divider() }
                ForEach(appState.allAccounts, id: \.self) { acc in
                    Button { toggle(acc) } label: {
                        Label(acc, systemImage: accounts.contains(acc) ? "checkmark" : "")
                    }
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
