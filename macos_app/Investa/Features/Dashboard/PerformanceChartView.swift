import SwiftUI
import Charts

struct PerformanceChartView: View {
    let points: [PerformancePoint]
    let currency: String
    let benchmarks: [String]
    @Binding var period: Period
    @Binding var customFrom: Date
    @Binding var customTo: Date

    @State private var view: PerformanceView = .value

    private struct SeriesPoint: Identifiable {
        let id = UUID()
        let date: Date
        let value: Double
        let series: String
    }

    private var seriesData: [SeriesPoint] {
        var out: [SeriesPoint] = []
        // Normalize TWR relative to the first point in the period (mirrors the
        // web PerformanceGraph), so the line starts at 0 and its end equals the
        // period TWR shown in the header.
        let baseFactor = 1 + (points.first(where: { $0.twr != nil })?.twr ?? 0) / 100
        for p in points {
            guard let d = p.parsedDate else { continue }
            switch view {
            case .value:
                out.append(SeriesPoint(date: d, value: p.value, series: "Portfolio"))
            case .twr:
                if let t = p.twr {
                    let adj = baseFactor != 0 ? ((1 + t / 100) / baseFactor - 1) * 100 : t
                    out.append(SeriesPoint(date: d, value: adj, series: "Portfolio"))
                }
                for b in benchmarks {
                    if let v = p.benchmark(b) { out.append(SeriesPoint(date: d, value: v, series: b)) }
                }
            case .drawdown:
                out.append(SeriesPoint(date: d, value: p.drawdown ?? 0, series: "Portfolio"))
            }
        }
        
        if period == .oneDay, let lastD = out.last?.date {
            var cal = Calendar(identifier: .gregorian)
            if let tz = TimeZone(identifier: "America/New_York") {
                cal.timeZone = tz
                let comps = cal.dateComponents([.year, .month, .day], from: lastD)
                if let start = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 9, minute: 30)),
                   let end = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 16, minute: 0)) {
                    out = out.filter { $0.date >= start && $0.date <= end }
                }
            }
        }
        
        return out
    }

    /// Headline stats for the selected period, matching the web PerformanceGraph:
    /// the value change (currency + %) for the Value view, and the period TWR
    /// (plus annualized when over a year) for the TWR view.
    private struct PeriodStat: Identifiable { let id = UUID(); let label: String; let text: String; let positive: Bool }

    private func pctStr(_ v: Double) -> String { String(format: "%.2f%%", v) }

    private var periodStats: [PeriodStat]? {
        guard points.count >= 2 else { return nil }
        switch view {
        case .value:
            guard let startVal = points.first?.value, let endVal = points.last?.value else { return nil }
            let change = endVal - startVal
            let pct = startVal != 0 ? change / startVal * 100 : 0
            return [PeriodStat(label: "Period Change",
                               text: "\(Fmt.currency(change, code: currency)) (\(pctStr(pct)))",
                               positive: change >= 0)]
        case .twr:
            let twrs = points.compactMap(\.twr)
            guard let startTwr = twrs.first, let endTwr = twrs.last else { return nil }
            let baseFactor = 1 + startTwr / 100
            let periodTwr = baseFactor != 0 ? ((1 + endTwr / 100) / baseFactor - 1) * 100 : endTwr
            var stats = [PeriodStat(label: "Period TWR", text: pctStr(periodTwr), positive: periodTwr >= 0)]
            if let sd = points.first?.parsedDate, let ed = points.last?.parsedDate {
                let years = ed.timeIntervalSince(sd) / (365.25 * 86400)
                if years > 1 {
                    let ann = (pow(1 + periodTwr / 100, 1 / years) - 1) * 100
                    stats.append(PeriodStat(label: "Ann. TWR", text: pctStr(ann), positive: ann >= 0))
                }
            }
            return stats
        case .drawdown:
            return nil
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 7) {
                    Image(systemName: "chart.xyaxis.line").font(.caption.weight(.semibold)).foregroundStyle(Theme.brand)
                    Text("Performance").font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                if let stats = periodStats {
                    HStack(spacing: 16) {
                        ForEach(stats) { s in
                            HStack(spacing: 6) {
                                Text(s.label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
                                Text(s.text).font(.callout.weight(.bold))
                                    .foregroundStyle(s.positive ? .green : .red)
                                    .lineLimit(1).minimumScaleFactor(0.7)
                            }
                        }
                        Spacer(minLength: 0)
                    }
                }
                Picker("View", selection: $view) {
                    ForEach(PerformanceView.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
            }
            Divider()

            periodRow
            if period == .custom { customDateRow }

            if seriesData.isEmpty {
                ContentUnavailableView("No history", systemImage: "chart.xyaxis.line")
                    .frame(height: 240)
            } else {
                chart
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private var periodRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Period.allCases) { p in
                    Button { period = p } label: {
                        Text(p.label).font(.caption.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background(period == p ? Color.accentColor : Color.gray.opacity(0.15), in: Capsule())
                            .foregroundStyle(period == p ? .white : .secondary)
                    }.buttonStyle(.plain)
                }
            }.padding(.vertical, 1)
        }
    }

    private var customDateRow: some View {
        HStack(spacing: 8) {
            Text("FROM").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customFrom, in: ...customTo, displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize().gregorianCalendar()
            Text("TO").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customTo, in: customFrom...Date(), displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize().gregorianCalendar()
            Spacer()
        }
        .padding(.vertical, 2)
    }
    @ViewBuilder private var chart: some View {
        let domain = chartDomain(seriesData.map(\.value))
        let baseChart = Chart(seriesData) { item in
            if view == .value {
                // Bound the fill to the visible domain; an implicit 0 baseline
                // sits far below the domain min and spills below the x-axis.
                AreaMark(x: .value("Date", item.date),
                         yStart: .value("Min", domain.lowerBound),
                         yEnd: .value("Value", item.value))
                    .foregroundStyle(
                        .linearGradient(colors: [Color.accentColor.opacity(0.30), Color.accentColor.opacity(0.02)],
                                        startPoint: .top, endPoint: .bottom))
            } else if view == .drawdown {
                AreaMark(x: .value("Date", item.date), y: .value("Value", item.value))
                    .foregroundStyle(
                        .linearGradient(colors: [Color.red.opacity(0.30), Color.red.opacity(0.02)],
                                        startPoint: .top, endPoint: .bottom))
            }
            LineMark(x: .value("Date", item.date), y: .value("Value", item.value))
                .foregroundStyle(by: .value("Series", item.series))
                .interpolationMethod(.monotone)
        }
        .chartForegroundStyleScale(range: seriesColors)
        .chartYScale(domain: domain)
        .chartLegend(view == .twr && !benchmarks.isEmpty ? .visible : .hidden)
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(view == .value ? valueAxisLabel(v, domain) : String(format: "%.0f%%", v))
                    }
                }
            }
        }
        .chartHoverTooltip(distinctDates) { i in
            let date = distinctDates[i]
            let entries = seriesData.filter { $0.date == date }
            guard !entries.isEmpty else { return nil }
            return ChartTooltipContent(title: tooltipString(date), rows: entries.map {
                ChartTooltipRow(color: seriesColor($0.series), label: $0.series,
                                value: view == .value ? Fmt.currency($0.value, code: currency)
                                                      : String(format: "%.2f%%", $0.value))
            })
        }
        
        baseChart.frame(height: 260)
    }

    /// Y-axis label for the Value view: in millions, with just enough decimals
    /// that adjacent ticks read differently (derived from the visible range).
    private func valueAxisLabel(_ v: Double, _ domain: ClosedRange<Double>) -> String {
        let rangeM = (domain.upperBound - domain.lowerBound) / 1_000_000
        let stepM = max(rangeM / 5, 1e-9)   // ~5 ticks
        let decimals = min(3, max(0, Int(ceil(-log10(stepM)))))
        return "\(Fmt.symbol(currency))\(String(format: "%.\(decimals)f", v / 1_000_000))M"
    }

    private var distinctDates: [Date] {
        var seen = Set<Date>(); var out: [Date] = []
        for p in seriesData where !seen.contains(p.date) { seen.insert(p.date); out.append(p.date) }
        return out
    }

    private func seriesColor(_ name: String) -> Color {
        if name == "Portfolio" { return view == .drawdown ? .red : .accentColor }
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink, .teal]
        if let idx = benchmarks.firstIndex(of: name) { return palette[idx % palette.count] }
        return .secondary
    }

    private func tooltipString(_ d: Date) -> String {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        if period == .oneDay || period == .fiveDays {
            f.timeZone = TimeZone(identifier: "America/New_York"); f.dateFormat = "EEE, MMM d h:mm a"
        } else {
            f.dateStyle = .medium
        }
        return f.string(from: d)
    }

    private var seriesColors: [Color] {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink, .teal]
        switch view {
        case .value: return [.accentColor]
        case .drawdown: return [.red]
        case .twr:
            return [.accentColor] + Array(palette.prefix(benchmarks.count))
        }
    }
}
