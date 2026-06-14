import SwiftUI
import Charts

struct PerformanceChartView: View {
    let points: [PerformancePoint]
    let currency: String

    private var dated: [(date: Date, value: Double)] {
        points.compactMap { p in
            guard let d = p.parsedDate else { return nil }
            return (d, p.value)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Portfolio Value")
                .font(.headline)

            if dated.isEmpty {
                ContentUnavailableView("No history", systemImage: "chart.xyaxis.line")
                    .frame(height: 240)
            } else {
                Chart(dated, id: \.date) { item in
                    AreaMark(
                        x: .value("Date", item.date),
                        y: .value("Value", item.value)
                    )
                    .foregroundStyle(
                        .linearGradient(
                            colors: [.accentColor.opacity(0.35), .accentColor.opacity(0.02)],
                            startPoint: .top, endPoint: .bottom
                        )
                    )
                    LineMark(
                        x: .value("Date", item.date),
                        y: .value("Value", item.value)
                    )
                    .foregroundStyle(.tint)
                    .interpolationMethod(.monotone)
                }
                .chartYAxis {
                    AxisMarks { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(Fmt.currency(v, code: currency))
                            }
                        }
                    }
                }
                .frame(height: 240)
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1)
        )
    }
}
