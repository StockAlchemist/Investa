import SwiftUI
import Charts

/// The financial-statement chart — the SwiftUI twin of
/// `web_app/lib/statement_chart.ts` and the Financials tab it feeds. Which line
/// items open, which colour each one wears, which of them can honestly share a
/// y-axis, and how a period end is named.

enum StatementPeriod: String, CaseIterable, Identifiable, Sendable {
    case quarterly, annual
    var id: String { rawValue }
    var title: String { self == .quarterly ? "Quarterly" : "Annual" }

    /// Quarters open on five years; a year per bar reads well over ten.
    var defaultRange: StatementRange { self == .quarterly ? .fiveYears : .tenYears }
}

/// How far back the chart plots. A range is a span in years — the unit an
/// investor thinks in — so it means a different column count per period type.
enum StatementRange: String, CaseIterable, Identifiable {
    case fiveYears = "5Y", tenYears = "10Y", max = "MAX"
    var id: String { rawValue }

    func periods(_ periodType: StatementPeriod) -> Int {
        switch self {
        case .max: return StatementChartConfig.maxPeriods
        case .fiveYears: return periodType == .quarterly ? 20 : 5
        case .tenYears: return periodType == .quarterly ? 40 : 10
        }
    }
}

enum StatementChartConfig {
    /// What the chart opens on for each statement, in preference order.
    static let defaultMetrics: [String: [String]] = [
        "income": ["Total Revenue", "Gross Profit", "Net Income"],
        "balance": ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"],
        "cash": ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"],
        "equity": ["Stockholders Equity", "Retained Earnings"],
    ]

    /// Four is where a validated categorical palette stops being separable.
    static let maxSeries = 4

    /// A hard ceiling on plotted periods, whatever range is asked for.
    static let maxPeriods = 80

    /// Past this many periods, grouped bars become hairlines and the chart is
    /// read as a shape rather than a set of magnitudes — a line's job.
    static let barToLineThreshold = 24

    /// A line item has to be reported across the history to be worth opening
    /// on. Meta never tags gross profit, so it exists only where Yahoo derives
    /// it; charted by default it is an empty series beside two full ones.
    static let minDefaultCoverage = 0.5

    /// Two measures whose magnitudes differ by more than this get their own
    /// chart rather than a second y-axis — revenue and EPS on one scale is a
    /// flat line next to a mountain, and on two scales it is a lie about their
    /// relationship.
    ///
    /// Ten means the smaller series still reaches a tenth of the frame, which
    /// is about where a bar stops being a readable magnitude and becomes a mark
    /// that something was there.
    static let sameScaleRatio = 10.0

    /// The first four slots of the validated categorical palette, stepped for
    /// each surface. Same hexes as the web app, so a line item is the same
    /// colour on every client.
    static func colors(_ scheme: ColorScheme) -> [Color] {
        scheme == .dark
            ? [Color(hex: 0x3987E5), Color(hex: 0xD95926), Color(hex: 0x199E70), Color(hex: 0xC98500)]
            : [Color(hex: 0x2A78D6), Color(hex: 0xEB6834), Color(hex: 0x1BAF7A), Color(hex: 0xEDA100)]
    }
}

/// How a period end is named on an axis or a column head. A quarter needs its
/// month; a year does not, because filed period ends are the company's own
/// 52/53-week dates and two of them can land in one calendar year.
func statementPeriodLabel(_ iso: String, _ period: StatementPeriod) -> String {
    period == .quarterly ? MarketTime.monthYear(iso) : String(iso.prefix(4))
}

/// Compact for statement magnitudes, plain for per-share figures.
func formatStatementValue(_ v: Double) -> String {
    let a = abs(v)
    if a >= 1_000_000_000_000 { return String(format: "%.2fT", v / 1_000_000_000_000) }
    if a >= 1_000_000_000 { return String(format: "%.2fB", v / 1_000_000_000) }
    if a >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
    if a >= 1_000 { return String(format: "%.2fK", v / 1_000) }
    return Fmt.number(v, fractionDigits: abs(v.rounded() - v) < 0.005 ? 0 : 2)
}

/// The line items to open on: the statement's preferred ones, less any too
/// sparsely reported to plot. Falls back to the preferred list when that would
/// leave nothing — a company that tags almost nothing still gets a chart.
func pickDefaultMetrics(
    _ preferred: [String],
    _ rows: [StatementRow],
    limit: Int = StatementChartConfig.maxSeries
) -> [String] {
    let byLabel = Dictionary(rows.map { ($0.label, $0.values) }, uniquingKeysWith: { a, _ in a })
    let available = preferred.filter { byLabel[$0] != nil }
    let covered = available.filter { label in
        guard let values = byLabel[label], !values.isEmpty else { return false }
        let reported = values.compactMap { $0 }.count
        return Double(reported) / Double(values.count) >= StatementChartConfig.minDefaultCoverage
    }
    return Array((covered.isEmpty ? available : covered).prefix(limit))
}

/// Add or remove a line item, holding every other one in the colour slot it
/// already had: removing series 1 of 3 must not repaint series 2 and 3, so a
/// dropped item leaves a hole rather than closing the array up.
func toggleStatementSlot(_ slots: [String?], _ label: String, max: Int = StatementChartConfig.maxSeries) -> [String?] {
    var next = slots
    if let at = next.firstIndex(of: label) {
        next[at] = nil
        return next
    }
    if let free = next.firstIndex(where: { $0 == nil }) {
        next[free] = label
        return next
    }
    if next.count < max { next.append(label) }
    return next
}

/// Split series into sets that can honestly share one y-axis. Never a second
/// axis: two scales on one frame let the author decide where the lines cross.
func groupBySharedScale<T>(_ series: [T], maxAbs: (T) -> Double) -> [[T]] {
    var groups: [[T]] = []
    for s in series.sorted(by: { maxAbs($0) > maxAbs($1) }) {
        let mine = maxAbs(s)
        if let idx = groups.firstIndex(where: { group in
            let lead = maxAbs(group[0])
            // Two all-zero rows share a frame; a zero row never joins a real one.
            if lead == 0 || mine == 0 { return lead == 0 && mine == 0 }
            return lead <= mine * StatementChartConfig.sameScaleRatio
        }) {
            groups[idx].append(s)
        } else {
            groups.append([s])
        }
    }
    return groups
}

// MARK: - View models for the chart

/// One line item of a statement: its label and its value per period, newest
/// first, the way the API ships them.
struct StatementRow: Identifiable, Equatable {
    let label: String
    let values: [Double?]
    var id: String { label }
    /// A row of nothing but blanks can be listed but not plotted.
    var isChartable: Bool { values.contains { $0 != nil } }
}

/// One charted line item: its values in plot order (oldest first) and the
/// palette slot it holds.
struct StatementSeries: Identifiable, Equatable {
    let slot: Int
    let label: String
    let color: Color
    /// Oldest-first, aligned with the chart's periods; nil where unreported.
    let values: [Double?]
    var id: Int { slot }
    var maxAbs: Double { values.compactMap { $0 }.map(abs).max() ?? 0 }
}

/// One chart, one y-axis. Callers split series that cannot share a scale into
/// separate charts rather than reaching for a second axis.
struct StatementChartView: View {
    let periods: [String]
    let series: [StatementSeries]
    let periodType: StatementPeriod

    private var hasNegative: Bool { series.contains { $0.values.contains { ($0 ?? 0) < 0 } } }

    /// Grouped bars stop reading as magnitudes once they are hairlines: past
    /// two dozen periods the question is the shape of the series, which is a
    /// line's job. Fifteen years of quarters is a line chart either way.
    private var asLines: Bool { periods.count > StatementChartConfig.barToLineThreshold }

    /// Every fourth or fifth period end, so the axis carries about six labels.
    /// A categorical x-scale draws a mark for every category it is given, so
    /// thinning has to happen here — sixty quarter labels overprint into a
    /// smear, which no font size fixes.
    private var axisPeriods: [String] {
        guard periods.count > 6 else { return periods }
        let step = Int(ceil(Double(periods.count) / 6.0))
        // Anchored on the newest period: the right-hand end is the one a reader
        // looks for, and it must not be the label that gets dropped.
        return periods.enumerated()
            .filter { (periods.count - 1 - $0.offset) % step == 0 }
            .map(\.element)
    }

    /// Where a tick label hangs relative to its mark: inward at the two ends,
    /// centred everywhere else.
    private func edgeAnchor(_ iso: String?) -> UnitPoint {
        guard let iso else { return .top }
        if iso == periods.last { return .topTrailing }
        if iso == periods.first { return .topLeading }
        return .top
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // One series is named by its title; several need the legend, since
            // identity must never rest on colour alone.
            HStack(spacing: 14) {
                ForEach(series) { s in
                    HStack(spacing: 5) {
                        RoundedRectangle(cornerRadius: 2).fill(s.color).frame(width: 9, height: 9)
                        Text(s.label).font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }

            Chart {
                ForEach(series) { s in
                    // Plotted against the period end itself, not its label: two
                    // fiscal years can carry the same display year (Advance Auto
                    // Parts closed years on 2022-01-01 and 2022-12-31) and a
                    // categorical scale would merge them into one column.
                    ForEach(Array(periods.enumerated()), id: \.offset) { i, period in
                        if let v = s.values.indices.contains(i) ? s.values[i] : nil {
                            if asLines {
                                LineMark(x: .value("Period", period), y: .value("Value", v))
                                    .foregroundStyle(s.color)
                                    .lineStyle(.init(lineWidth: 2))
                                    .interpolationMethod(.monotone)
                                    .position(by: .value("Series", s.label))
                            } else {
                                // Wider than the default: the space between
                                // periods is worth less than a readable bar.
                                BarMark(
                                    x: .value("Period", period),
                                    y: .value("Value", v),
                                    width: .ratio(0.9)
                                )
                                    .foregroundStyle(s.color)
                                    .position(by: .value("Series", s.label))
                                    .cornerRadius(4)
                            }
                        }
                    }
                }
                if hasNegative {
                    RuleMark(y: .value("Zero", 0))
                        .foregroundStyle(.secondary.opacity(0.5))
                        .lineStyle(.init(lineWidth: 1))
                }
            }
            .chartForegroundStyleScale(domain: series.map(\.label), range: series.map(\.color))
            .chartLegend(.hidden)
            .chartXAxis {
                AxisMarks(values: axisPeriods) { value in
                    let iso = value.as(String.self)
                    // A label centred on the outermost category hangs half its
                    // width past the plot and is truncated there ("Jun 2…").
                    // The edge labels hang inward instead.
                    AxisValueLabel(anchor: edgeAnchor(iso)) {
                        if let iso {
                            Text(statementPeriodLabel(iso, periodType)).font(.caption2)
                        }
                    }
                }
            }
            .chartYAxis {
                // Leading, as the web app draws it: the same statement should
                // not read right-to-left on one client and left-to-right on
                // another, and it frees the trailing edge for the newest
                // period's x label.
                AxisMarks(position: .leading) { value in
                    AxisGridLine().foregroundStyle(.secondary.opacity(0.15))
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(formatStatementValue(v)).font(.caption2)
                        }
                    }
                }
            }
            .frame(height: 220)
            // The newest period's label is centred under a category sitting at
            // the very edge, so half of it would be clipped. Insetting the plot
            // by half a label leaves it somewhere to go.
            .chartPlotStyle { plot in plot.padding(.trailing, 30) }
            // One tooltip per period listing every series, so the pointer never
            // has to find a particular bar to read a number.
            .chartHoverTooltip(periods) { i in
                guard periods.indices.contains(i) else { return nil }
                return ChartTooltipContent(
                    title: MarketTime.formatted(periods[i]),
                    rows: series.map { s in
                        let v = s.values.indices.contains(i) ? s.values[i] : nil
                        return ChartTooltipRow(
                            color: s.color,
                            label: s.label,
                            value: v.map(formatStatementValue) ?? "—"
                        )
                    }
                )
            }
        }
    }
}

/// The three numbers a quarterly statement is usually opened for: where the
/// headline line item landed, how it moved on the prior period, and how it moved
/// on the same period a year earlier — the one comparison a seasonal business
/// can be judged on.
struct StatementChangeStrip: View {
    let series: StatementSeries
    let periods: [String]
    let periodType: StatementPeriod

    private var lastIndex: Int? {
        series.values.lastIndex(where: { $0 != nil })
    }

    /// A percentage change off a negative or zero base says nothing.
    private func change(_ latest: Double, _ back: Int) -> Double? {
        guard let last = lastIndex, last - back >= 0,
              let prior = series.values[last - back], prior > 0 else { return nil }
        return (latest - prior) / prior * 100
    }

    var body: some View {
        if let last = lastIndex, let latest = series.values[last] {
            let yearBack = periodType == .quarterly ? 4 : 1
            HStack(alignment: .top, spacing: 12) {
                cell(
                    title: "\(series.label) · \(periods.indices.contains(last) ? statementPeriodLabel(periods[last], periodType) : "")",
                    value: formatStatementValue(latest),
                    tone: nil,
                    dot: true
                )
                if periodType == .quarterly {
                    cell(title: "vs prior quarter", value: pct(change(latest, 1)), tone: change(latest, 1), dot: false)
                }
                cell(
                    title: periodType == .quarterly ? "vs year ago" : "vs prior year",
                    value: pct(change(latest, yearBack)),
                    tone: change(latest, yearBack),
                    dot: false
                )
            }
        }
    }

    private func pct(_ v: Double?) -> String {
        guard let v else { return "—" }
        return String(format: "%@%.1f%%", v >= 0 ? "+" : "", v)
    }

    private func cell(title: String, value: String, tone: Double?, dot: Bool) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 5) {
                if dot { Circle().fill(series.color).frame(width: 7, height: 7) }
                Text(title).font(.caption2.weight(.bold)).textCase(.uppercase)
                    .foregroundStyle(.secondary).lineLimit(1)
            }
            Text(value)
                .font(.title3.bold().monospacedDigit())
                .foregroundStyle(tone == nil ? Color.primary : (tone! >= 0 ? .green : .red))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
    }
}
