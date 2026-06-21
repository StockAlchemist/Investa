import SwiftUI
import Charts

/// Forward portfolio-value projection (mirrors the web `ProjectionCard`):
/// a fan chart of the median + 10–90 / 25–75 percentile bands over 1/3/5/10/20y,
/// plus a table. Driven by `GET /projection` (lognormal model).
struct ProjectionCardView: View {
    let projection: Projection?
    let currency: String
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var isPhone: Bool { hSize == .compact }
    #else
    private var isPhone: Bool { false }
    #endif

    /// Years tabulated below the chart (the chart plots every year).
    private static let milestones: Set<Int> = [1, 3, 5, 10, 20]

    private var cur: String { projection?.currency ?? currency }
    private var horizons: [ProjectionHorizon] { projection?.horizons ?? [] }
    private var milestoneHorizons: [ProjectionHorizon] { horizons.filter { Self.milestones.contains($0.years) } }

    /// Cap the y-axis at the final horizon's 75th percentile so the median and the
    /// likely (25–75%) band fill the height instead of being dwarfed by the
    /// extreme upper tail; the outer band simply extends past the top for the
    /// longest horizons.
    private var yMax: Double { horizons.last.map { max($0.p75, $0.medianValue) } ?? 1 }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()

            if let p = projection, p.available, !horizons.isEmpty {
                assumptions(p)
                chart
                table
                footnote
            } else {
                ContentUnavailableView(
                    "Not enough history",
                    systemImage: "chart.line.uptrend.xyaxis",
                    description: Text("Projections appear once the portfolio has a longer track record.")
                )
                .frame(height: 200)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private var header: some View {
        HStack(spacing: 7) {
            Image(systemName: "chart.line.uptrend.xyaxis").font(.caption.weight(.semibold)).foregroundStyle(Theme.brand)
            Text("Projected Value").font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase)
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private func assumptions(_ p: Projection) -> some View {
        let ret = stat("Assumed return", "\(Fmt.percent(p.annualReturnPct)) p.a.")
        let vol = stat("Volatility", String(format: "%.2f%%", p.annualVolatilityPct ?? 0))
        // Single line when there's room (landscape / iPad / macOS); stacked when
        // the width is too narrow (iPhone portrait) so nothing wraps or truncates.
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) { ret; vol }
            VStack(alignment: .leading, spacing: 4) { ret; vol }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
                .lineLimit(1)
            Text(value).font(.callout.weight(.bold)).lineLimit(1)
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private var chart: some View {
        Chart {
            // Each band is its own series, otherwise Swift Charts interleaves the
            // two bands' points into one zigzag (sawtooth) path.
            ForEach(horizons) { h in
                AreaMark(x: .value("Years", h.years),
                         yStart: .value("Low", h.p10), yEnd: .value("High", h.p90))
                    .foregroundStyle(by: .value("Band", "10–90%"))
                    .interpolationMethod(.monotone)
            }
            ForEach(horizons) { h in
                AreaMark(x: .value("Years", h.years),
                         yStart: .value("Low", h.p25), yEnd: .value("High", h.p75))
                    .foregroundStyle(by: .value("Band", "25–75%"))
                    .interpolationMethod(.monotone)
            }
            ForEach(horizons) { h in
                LineMark(x: .value("Years", h.years), y: .value("Median", h.medianValue))
                    .foregroundStyle(Theme.brand)
                    .interpolationMethod(.monotone)
            }
            ForEach(milestoneHorizons) { h in
                PointMark(x: .value("Years", h.years), y: .value("Median", h.medianValue))
                    .foregroundStyle(Theme.brand)
                    .symbolSize(24)
            }
        }
        .chartForegroundStyleScale([
            "10–90%": Theme.brand.opacity(0.12),
            "25–75%": Theme.brand.opacity(0.22),
        ])
        .chartLegend(.hidden)
        .chartYScale(domain: 0...yMax)
        .chartXScale(domain: 0...(Double(horizons.last?.years ?? 20)))
        .chartXAxis {
            AxisMarks(values: [1, 5, 10, 15, 20]) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let y = value.as(Int.self) { Text("\(y)Y") }
                }
            }
        }
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) { Text(Fmt.compact(v, code: cur)) }
                }
            }
        }
        .chartHoverTooltip(horizons.map(\.years)) { i in
            guard i < horizons.count else { return nil }
            let h = horizons[i]
            return ChartTooltipContent(title: "In \(h.years) \(h.years == 1 ? "year" : "years")", rows: [
                ChartTooltipRow(color: Theme.brand, label: "Median",
                                value: Fmt.currency(h.medianValue, code: cur)),
                ChartTooltipRow(label: "Expected", value: Fmt.currency(h.expectedValue, code: cur)),
                ChartTooltipRow(label: "Return", value: Fmt.percent(h.medianReturnPct)),
                ChartTooltipRow(label: "10–90%",
                                value: "\(Fmt.compact(h.p10, code: cur)) – \(Fmt.compact(h.p90, code: cur))"),
            ])
        }
        .frame(height: 220)
    }

    private var table: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Horizon").frame(maxWidth: .infinity, alignment: .leading)
                Text("Median").frame(maxWidth: .infinity, alignment: .trailing)
                Text("Return").frame(maxWidth: .infinity, alignment: .trailing)
                if !isPhone {
                    Text("Range (10–90%)").frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
            .font(.system(size: 10, weight: .bold)).textCase(.uppercase).tracking(0.5)
            .foregroundStyle(.secondary)
            .padding(.bottom, 6)

            ForEach(milestoneHorizons) { h in
                Divider()
                HStack {
                    Text("\(h.years) \(h.years == 1 ? "year" : "years")")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(Fmt.currency(h.medianValue, code: cur))
                        .fontWeight(.bold).monospacedDigit().lineLimit(1).minimumScaleFactor(0.6)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    Text(Fmt.percent(h.medianReturnPct))  // Fmt.percent already prefixes +/-
                        .fontWeight(.semibold).monospacedDigit()
                        .foregroundStyle(h.medianReturnPct >= 0 ? Color.green : .red)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    if !isPhone {
                        Text("\(Fmt.compact(h.p10, code: cur)) – \(Fmt.compact(h.p90, code: cur))")
                            .foregroundStyle(.secondary).monospacedDigit().lineLimit(1).minimumScaleFactor(0.6)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                }
                .font(.subheadline)
                .padding(.vertical, 8)
            }
        }
    }

    private var footnote: some View {
        Text("Compounds the portfolio's historical annualized return and volatility forward (lognormal model). The median is the central estimate; the band shows the 10th–90th percentile range. Past performance does not guarantee future results — longer horizons are far more uncertain.")
            .font(.caption2).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
