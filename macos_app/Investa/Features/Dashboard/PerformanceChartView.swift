import SwiftUI
import Charts

struct PerformanceChartView: View {
    let points: [PerformancePoint]
    let currency: String
    let benchmarks: [String]
    @Binding var period: Period

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
            HStack {
                Text("Performance").font(.headline)
                Spacer()
                Picker("View", selection: $view) {
                    ForEach(PerformanceView.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).fixedSize()
                Picker("Period", selection: $period) {
                    ForEach(Period.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented).fixedSize()
            }

            if seriesData.isEmpty {
                ContentUnavailableView("No history", systemImage: "chart.xyaxis.line")
                    .frame(height: 240)
            } else {
                chart
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1)
        )
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
        .frame(height: 260)
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
