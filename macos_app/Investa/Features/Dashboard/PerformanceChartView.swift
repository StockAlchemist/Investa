import SwiftUI
import Charts

struct PerformanceChartView: View {
    let points: [PerformancePoint]
    let currency: String
    let benchmarks: [String]
    @Binding var period: Period
    @Binding var customFrom: Date
    @Binding var customTo: Date
    var isLoading: Bool = false

    @State private var view: PerformanceView = .value

    private struct SeriesPoint: Identifiable {
        let date: Date
        let value: Double
        let series: String
        // Stable identity (vs. a fresh UUID each rebuild) so Swift Charts can diff
        // points across renders instead of tearing down and re-animating the whole
        // chart on every currency/view change.
        var id: String { "\(series)@\(date.timeIntervalSinceReferenceDate)" }
    }

    /// All chart-derived data, computed in a single pass per render. Building this
    /// once (rather than via several interdependent computed properties that each
    /// re-loop `points` and re-parse dates) is what keeps redraws — especially the
    /// currency switch — cheap.
    private struct ChartModel {
        let series: [SeriesPoint]
        let dates: [Date]
        let dIndices: [Date: Int]
        let domain: ClosedRange<Double>
        let fxPoints: [FXPoint]
        let fxDomain: ClosedRange<Double>
        let showFX: Bool
        let fxSeriesName: String
        var isEmpty: Bool { series.isEmpty }
    }

    private func buildModel() -> ChartModel {
        let fxName = "FX (\(currency)/USD)"
        let startFX = points.first(where: { $0.fxRate != nil })?.fxRate
        let showFX = currency.uppercased() != "USD" && startFX != nil

        // Series points (single pass). Normalize TWR relative to the first point
        // in the period (mirrors the web PerformanceGraph), so the line starts at
        // 0 and its end equals the period TWR shown in the header.
        let baseFactor = 1 + (points.first(where: { $0.twr != nil })?.twr ?? 0) / 100
        var series: [SeriesPoint] = []
        for p in points {
            guard let d = p.parsedDate else { continue }
            switch view {
            case .value:
                series.append(SeriesPoint(date: d, value: p.value, series: "Portfolio"))
            case .twr:
                if let t = p.twr {
                    let adj = baseFactor != 0 ? ((1 + t / 100) / baseFactor - 1) * 100 : t
                    series.append(SeriesPoint(date: d, value: adj, series: "Portfolio"))
                }
                for b in benchmarks {
                    if let v = p.benchmark(b) { series.append(SeriesPoint(date: d, value: v, series: b)) }
                }
                // FX overlay (return view): the FX rate's % change from the period
                // start, on the shared percentage axis, as a dashed line.
                if showFX, let r = p.fxRate, let s = startFX, s != 0 {
                    series.append(SeriesPoint(date: d, value: (r / s - 1) * 100, series: fxName))
                }
            case .drawdown:
                series.append(SeriesPoint(date: d, value: p.drawdown ?? 0, series: "Portfolio"))
            }
        }
        series = filterMarketHours(series) { $0.date }

        // Distinct x-values and their category indices, derived from `series`.
        var seen = Set<Date>(); var dates: [Date] = []
        for s in series where !seen.contains(s.date) { seen.insert(s.date); dates.append(s.date) }
        var dIndices: [Date: Int] = [:]
        for (i, d) in dates.enumerated() { dIndices[d] = i }

        // FX overlay points (single pass) for the value chart + tooltips.
        var fxPoints: [FXPoint] = []
        if showFX, let s = startFX, s != 0 {
            for p in points {
                guard let d = p.parsedDate, let r = p.fxRate else { continue }
                fxPoints.append(FXPoint(date: d, rate: r, ret: (r / s - 1) * 100))
            }
            fxPoints = filterMarketHours(fxPoints) { $0.date }
        }

        return ChartModel(
            series: series,
            dates: dates,
            dIndices: dIndices,
            domain: chartDomain(series.map(\.value)),
            fxPoints: fxPoints,
            fxDomain: chartDomain(fxPoints.map(\.rate), pad: 0.05),
            showFX: showFX,
            fxSeriesName: fxName
        )
    }

    /// The 1D chart only spans US market hours (9:30–16:00 ET); trim any points
    /// outside that window. Shared by every series (portfolio, benchmarks, FX).
    private func filterMarketHours<T>(_ items: [T], date: (T) -> Date) -> [T] {
        guard period == .oneDay, let lastD = items.last.map(date),
              let tz = TimeZone(identifier: "America/New_York") else { return items }
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = tz
        let comps = cal.dateComponents([.year, .month, .day], from: lastD)
        guard let start = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 9, minute: 30)),
              let end = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 16, minute: 0)) else { return items }
        return items.filter { date($0) >= start && date($0) <= end }
    }

    // MARK: - FX overlay

    /// The FX line's series name / legend label, e.g. "FX (THB/USD)". Cheap
    /// (O(1)) — safe to reference from per-point styling closures.
    private var fxSeriesName: String { "FX (\(currency)/USD)" }

    /// FX multiplier at the first point that carries one — the FX-return baseline.
    private var startFX: Double? { points.first(where: { $0.fxRate != nil })?.fxRate }

    /// Show the FX overlay only when the display currency isn't USD and the
    /// backend actually returned historical FX rates. Referenced O(1) times per
    /// render (legend, colors, view guards) — never inside a per-point loop.
    private var showFX: Bool { currency.uppercased() != "USD" && startFX != nil }

    private struct FXPoint: Identifiable {
        let date: Date
        let rate: Double
        let ret: Double
        var id: TimeInterval { date.timeIntervalSinceReferenceDate }
    }

    /// Map an FX rate into the primary (value) axis coordinate space so the FX
    /// line can share the value chart, and its inverse for the right-axis labels.
    /// Both take the precomputed domains — never recompute them here (hot path).
    private func fxToValueY(_ rate: Double, valueDomain: ClosedRange<Double>, fxDomain: ClosedRange<Double>) -> Double {
        guard fxDomain.upperBound > fxDomain.lowerBound else { return valueDomain.lowerBound }
        let t = (rate - fxDomain.lowerBound) / (fxDomain.upperBound - fxDomain.lowerBound)
        return valueDomain.lowerBound + t * (valueDomain.upperBound - valueDomain.lowerBound)
    }
    private func valueYToFX(_ y: Double, valueDomain: ClosedRange<Double>, fxDomain: ClosedRange<Double>) -> Double {
        guard valueDomain.upperBound > valueDomain.lowerBound else { return fxDomain.lowerBound }
        let t = (y - valueDomain.lowerBound) / (valueDomain.upperBound - valueDomain.lowerBound)
        return fxDomain.lowerBound + t * (fxDomain.upperBound - fxDomain.lowerBound)
    }
    /// Value-axis positions for the five evenly spaced FX ticks on the right axis.
    private func fxAxisPositions(valueDomain: ClosedRange<Double>, fxDomain: ClosedRange<Double>) -> [Double] {
        let n = 5
        return (0..<n).map { i in
            let rate = fxDomain.lowerBound + Double(i) / Double(n - 1) * (fxDomain.upperBound - fxDomain.lowerBound)
            return fxToValueY(rate, valueDomain: valueDomain, fxDomain: fxDomain)
        }
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
        let model = buildModel()
        return VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 7) {
                    Image(systemName: "chart.xyaxis.line").font(.caption.weight(.semibold)).foregroundStyle(Theme.brand)
                    Text("Performance").font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase)
                        .foregroundStyle(.secondary)
                    if isLoading {
                        ProgressView().controlSize(.small)
                    }
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

            if isLoading && model.isEmpty {
                ProgressView()
                    .frame(height: 240)
                    .frame(maxWidth: .infinity)
            } else if model.isEmpty {
                ContentUnavailableView("No history", systemImage: "chart.xyaxis.line")
                    .frame(height: 240)
            } else {
                chart(model)
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
    @ViewBuilder private func chart(_ model: ChartModel) -> some View {
        let domain = model.domain
        let dates = model.dates
        let dIndices = model.dIndices
        let fxDomain = model.fxDomain

        Group {
            if period == .oneDay {
                Chart {
                    ForEach(model.series) { item in
                        if view == .value {
                            AreaMark(x: .value("Date", item.date), yStart: .value("Min", domain.lowerBound), yEnd: .value("Value", item.value))
                                .foregroundStyle(.linearGradient(colors: [Color.accentColor.opacity(0.30), Color.accentColor.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                        } else if view == .drawdown {
                            AreaMark(x: .value("Date", item.date), y: .value("Value", item.value))
                                .foregroundStyle(.linearGradient(colors: [Color.red.opacity(0.30), Color.red.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                        }
                        LineMark(x: .value("Date", item.date), y: .value("Value", item.value))
                            .foregroundStyle(by: .value("Series", item.series))
                            .lineStyle(seriesStroke(item.series))
                            .interpolationMethod(.monotone)
                    }
                    if view == .value, model.showFX {
                        ForEach(model.fxPoints) { fp in
                            LineMark(x: .value("Date", fp.date), y: .value("FX", fxToValueY(fp.rate, valueDomain: domain, fxDomain: fxDomain)))
                                .foregroundStyle(Theme.fx)
                                .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 5]))
                                .interpolationMethod(.monotone)
                        }
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { value in
                        if let date = value.as(Date.self) {
                            AxisValueLabel { Text(xAxisLabel(for: date, period: period)) }
                        }
                    }
                }
                .chartHoverTooltip(dates) { i in tooltipContent(for: dates[i], model: model) }
            } else {
                Chart {
                    ForEach(model.series) { item in
                        let xIdx = dIndices[item.date] ?? 0
                        if view == .value {
                            AreaMark(x: .value("Index", xIdx), yStart: .value("Min", domain.lowerBound), yEnd: .value("Value", item.value))
                                .foregroundStyle(.linearGradient(colors: [Color.accentColor.opacity(0.30), Color.accentColor.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                        } else if view == .drawdown {
                            AreaMark(x: .value("Index", xIdx), y: .value("Value", item.value))
                                .foregroundStyle(.linearGradient(colors: [Color.red.opacity(0.30), Color.red.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                        }
                        LineMark(x: .value("Index", xIdx), y: .value("Value", item.value))
                            .foregroundStyle(by: .value("Series", item.series))
                            .lineStyle(seriesStroke(item.series))
                            .interpolationMethod(.monotone)
                    }
                    if view == .value, model.showFX {
                        ForEach(model.fxPoints) { fp in
                            LineMark(x: .value("Index", dIndices[fp.date] ?? 0), y: .value("FX", fxToValueY(fp.rate, valueDomain: domain, fxDomain: fxDomain)))
                                .foregroundStyle(Theme.fx)
                                .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 5]))
                                .interpolationMethod(.monotone)
                        }
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { value in
                        if let idx = value.as(Int.self), idx >= 0, idx < dates.count {
                            AxisValueLabel { Text(xAxisLabel(for: dates[idx], period: period)) }
                        }
                    }
                }
                .chartXScale(domain: 0...max(0, dates.count - 1))
                .chartHoverTooltip(Array(dates.indices)) { i in tooltipContent(for: dates[i], model: model) }
            }
        }
        .chartForegroundStyleScale(range: seriesColors)
        .chartYScale(domain: domain)
        .chartLegend(view == .twr && (!benchmarks.isEmpty || showFX) ? .visible : .hidden)
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(formatAxis(v, domain: domain, isPercent: view != .value))
                    }
                }
            }
            // Secondary (right) FX-rate axis for the value view, matching the web.
            if view == .value, model.showFX {
                AxisMarks(position: .trailing, values: fxAxisPositions(valueDomain: domain, fxDomain: fxDomain)) { value in
                    if let y = value.as(Double.self) {
                        AxisValueLabel {
                            Text(String(format: "%.2f", valueYToFX(y, valueDomain: domain, fxDomain: fxDomain)))
                                .foregroundStyle(Theme.fx)
                        }
                    }
                }
            }
        }
        .frame(height: 260)
    }

    /// Dash the FX line; every other series stays solid.
    private func seriesStroke(_ series: String) -> StrokeStyle {
        series == fxSeriesName ? StrokeStyle(lineWidth: 1.5, dash: [5, 5]) : StrokeStyle(lineWidth: 2)
    }

    /// Tooltip card for the hovered date: portfolio/benchmark rows plus the FX
    /// rate and its period return, mirroring the web PerformanceGraph tooltip.
    private func tooltipContent(for date: Date, model: ChartModel) -> ChartTooltipContent? {
        let entries = model.series.filter { $0.date == date && $0.series != model.fxSeriesName }
        var rows = entries.map {
            ChartTooltipRow(color: seriesColor($0.series), label: $0.series,
                            value: view == .value ? Fmt.currency($0.value, code: currency) : String(format: "%.2f%%", $0.value))
        }
        if model.showFX, let fp = model.fxPoints.first(where: { $0.date == date }) {
            rows.append(ChartTooltipRow(color: Theme.fx, label: "FX Rate", value: String(format: "%.4f", fp.rate)))
            rows.append(ChartTooltipRow(color: Theme.fx, label: "FX Ret", value: String(format: "%.2f%%", fp.ret)))
        }
        guard !rows.isEmpty else { return nil }
        return ChartTooltipContent(title: tooltipString(date), rows: rows)
    }

    /// Y-axis label: dynamically formats K, M, B for both value and percent
    private func formatAxis(_ v: Double, domain: ClosedRange<Double>, isPercent: Bool) -> String {
        let maxMagnitude = max(abs(domain.upperBound), abs(domain.lowerBound))
        let divisor: Double
        let suffix: String
        
        if maxMagnitude >= 1_000_000_000 {
            divisor = 1_000_000_000
            suffix = "B"
        } else if maxMagnitude >= 1_000_000 {
            divisor = 1_000_000
            suffix = "M"
        } else if maxMagnitude >= 1_000 {
            divisor = 1_000
            suffix = "K"
        } else {
            divisor = 1
            suffix = ""
        }
        
        let range = (domain.upperBound - domain.lowerBound) / divisor
        let step = max(range / 5, 1e-9)   // ~5 ticks
        let decimals = min(3, max(0, Int(ceil(-log10(step)))))
        
        let formattedValue = String(format: "%.\(decimals)f", v / divisor)
        if isPercent {
            return "\(formattedValue)\(suffix)%"
        } else {
            return "\(Fmt.symbol(currency))\(formattedValue)\(suffix)"
        }
    }

    private func seriesColor(_ name: String) -> Color {
        if name == "Portfolio" { return view == .drawdown ? .red : .accentColor }
        if name == fxSeriesName { return Theme.fx }
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

    private func xAxisLabel(for d: Date, period: Period) -> String {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        if period == .oneDay {
            f.timeZone = TimeZone(identifier: "America/New_York")
            f.dateFormat = "h:mm a"
        } else if period == .fiveDays {
            f.timeZone = TimeZone(identifier: "America/New_York")
            f.dateFormat = "E"
        } else if period == .oneMonth {
            f.dateFormat = "MMM d"
        } else if period == .oneYear || period == .ytd {
            f.dateFormat = "MMM"
        } else {
            f.dateFormat = "yyyy"
        }
        return f.string(from: d)
    }

    private var seriesColors: [Color] {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink, .teal]
        switch view {
        case .value: return [.accentColor]
        case .drawdown: return [.red]
        case .twr:
            // Order must match series first-appearance in seriesData: Portfolio,
            // benchmarks, then the FX return line.
            return [.accentColor] + Array(palette.prefix(benchmarks.count)) + (showFX ? [Theme.fx] : [])
        }
    }
}
