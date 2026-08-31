import SwiftUI
import Charts

/// One measure plotted across a company's filed period ends — a ratio, a
/// margin, a share count.
///
/// This drawing existed twice, once in the Ratios & Trends grid and once in the
/// Key Metrics cards, and both copies had the same two faults: a constant
/// number of axis labels, and bars whatever the density. Nineteen filed years
/// of quarters is 76 bars, which on a phone is 3pt of ink each and reads as
/// noise, under four axis labels that overprint into one word. Both are
/// decisions about the room the chart got, so both are `PeriodChartMetrics`'.
struct FiledPeriodChart: View {
    struct Point: Identifiable {
        let period: Date
        let iso: String
        let value: Double
        var id: Date { period }
    }

    let points: [Point]
    let color: Color
    let periodType: StatementPeriod
    var height: CGFloat = 200
    /// Names the measure in the tooltip.
    let label: String
    /// Renders a value for the y axis and the tooltip, so both say it the same
    /// way — a raw share count is unreadable on an axis; 15.00B is the same
    /// number said usefully.
    let format: (Double) -> String

    @State private var chartWidth: CGFloat = 0
    @Environment(\.appFontScale) private var fontScale
    @Environment(\.dynamicTypeSize) private var typeSize

    private var metrics: PeriodChartMetrics {
        PeriodChartMetrics(
            containerWidth: chartWidth,
            periodCount: points.count,
            periodType: periodType,
            textScale: ChartAxis.textScale(scale: fontScale, typeSize: typeSize)
        )
    }

    /// A ratio can be negative (a loss-making margin), so the domain has to
    /// keep zero in frame rather than starting there.
    private var yDomain: ClosedRange<Double>? { periodChartDomain(points.map(\.value)) }

    private var axisDates: [Date] { metrics.thinned(points.map(\.period)) }

    var body: some View {
        chart
            .frame(height: height)
            // Must be the *offered* width: measuring what the chart resolved to
            // would latch a wide label count on. See `readingContainerWidth`.
            .readingContainerWidth { chartWidth = $0 }
    }

    private var chart: some View {
        Chart {
            ForEach(points) { p in
                if metrics.preferLines {
                    LineMark(x: .value("Period", p.period), y: .value(label, p.value))
                        .foregroundStyle(color)
                        .lineStyle(.init(lineWidth: 2))
                        .interpolationMethod(.monotone)
                    AreaMark(x: .value("Period", p.period), y: .value(label, p.value))
                        .foregroundStyle(.linearGradient(
                            colors: [color.opacity(0.22), color.opacity(0.02)],
                            startPoint: .top, endPoint: .bottom
                        ))
                } else {
                    BarMark(x: .value("Period", p.period), y: .value(label, p.value))
                        .foregroundStyle(color.gradient)
                        .cornerRadius(3)
                }
            }
            // A measure that crosses zero needs the crossing drawn; one that
            // never does would just get a second axis line.
            if points.contains(where: { $0.value < 0 }) {
                RuleMark(y: .value("Zero", 0))
                    .foregroundStyle(.secondary.opacity(0.5))
                    .lineStyle(.init(lineWidth: 1))
            }
        }
        .modifier(BoundedYScale(domain: yDomain))
        .chartXAxis {
            // Plotted against the period end itself, not a year string: two
            // fiscal years can end in the same calendar year and as categories
            // they would collapse onto one point.
            AxisMarks(values: axisDates) { value in
                AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                if let date = value.as(Date.self) {
                    AxisValueLabel {
                        // Four quarters a year would all read "2026".
                        Text(periodType == .quarterly
                             ? MarketTime.monthYear(date)
                             : MarketTime.year(date))
                            .appFont(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize()
                    }
                }
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: metrics.yTickCount)) { value in
                AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                if let v = value.as(Double.self) {
                    AxisValueLabel {
                        Text(format(v)).appFont(.caption2).foregroundStyle(.secondary).fixedSize()
                    }
                }
            }
        }
        // Most periods go unlabelled by design; the tooltip is how a particular
        // one is read off.
        .chartHoverTooltip(points.map(\.period)) { i in
            guard points.indices.contains(i) else { return nil }
            let p = points[i]
            return ChartTooltipContent(
                title: MarketTime.formatted(p.iso),
                rows: [ChartTooltipRow(color: color, label: label, value: format(p.value))]
            )
        }
    }
}

extension FiledPeriodChart {
    /// Builds the point list from the ratio-history rows the API ships, which
    /// arrive newest-first and carry their period end as an ISO calendar day.
    static func points(_ history: [[String: JSONValue]], key: String) -> [Point] {
        history.reversed().compactMap { item in
            guard let value = item[key]?.doubleValue,
                  let iso = item["Period"]?.stringValue,
                  let period = MarketTime.calendarDay(iso) else { return nil }
            return Point(period: period, iso: iso, value: value)
        }
    }
}
