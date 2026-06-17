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
        for p in points {
            guard let d = p.parsedDate else { continue }
            switch view {
            case .value:
                out.append(SeriesPoint(date: d, value: p.value, series: "Portfolio"))
            case .twr:
                if let t = p.twr { out.append(SeriesPoint(date: d, value: t, series: "Portfolio")) }
                for b in benchmarks {
                    if let v = p.benchmark(b) { out.append(SeriesPoint(date: d, value: v, series: b)) }
                }
            case .drawdown:
                out.append(SeriesPoint(date: d, value: p.drawdown ?? 0, series: "Portfolio"))
            }
        }
        return out
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
            Text("FROM").font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customFrom, in: ...customTo, displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize()
            Text("TO").font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customTo, in: customFrom...Date(), displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize()
            Spacer()
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private var chart: some View {
        Chart(seriesData) { item in
            if view == .value || view == .drawdown {
                AreaMark(x: .value("Date", item.date), y: .value("Value", item.value))
                    .foregroundStyle(
                        .linearGradient(
                            colors: [(view == .drawdown ? Color.red : .accentColor).opacity(0.30),
                                     (view == .drawdown ? Color.red : .accentColor).opacity(0.02)],
                            startPoint: .top, endPoint: .bottom))
            }
            LineMark(x: .value("Date", item.date), y: .value("Value", item.value))
                .foregroundStyle(by: .value("Series", item.series))
                .interpolationMethod(.monotone)
        }
        .chartForegroundStyleScale(range: seriesColors)
        .chartYScale(domain: chartDomain(seriesData.map(\.value)))
        .chartLegend(view == .twr && !benchmarks.isEmpty ? .visible : .hidden)
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(view == .value ? Fmt.currency(v, code: currency) : String(format: "%.0f%%", v))
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
        .frame(height: 260)
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
