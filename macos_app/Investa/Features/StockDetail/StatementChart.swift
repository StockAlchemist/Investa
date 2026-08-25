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

/// How a chart plotted over filed period ends should be sized for the room it
/// actually got.
///
/// Three charts in the app draw the same thing — the statement trend, the ratio
/// grid on the stock page, and the ratio cards under Key Metrics — and each had
/// invented its own answer, two of them a constant. A constant is always wrong
/// somewhere: five axis labels is right on a desktop and a smear on a phone,
/// and a bar per quarter is a magnitude at 900pt and a hairline at 300.
///
/// Approximate on purpose. These pick label counts and mark types, never a
/// frame, so a measurement that lags a layout pass can only mis-round.
struct PeriodChartMetrics {
    /// The width the container offered the chart.
    let containerWidth: CGFloat
    let periodCount: Int
    /// Series drawn side by side at each period; 1 for a single-series chart.
    var seriesCount: Int = 1
    let periodType: StatementPeriod

    /// What the y labels and the trailing inset cost before any mark is drawn.
    private static let axisOverhead: CGFloat = 90

    /// The room left for marks.
    var plotWidth: CGFloat { max(0, containerWidth - Self.axisOverhead) }

    /// Roughly what one period end occupies at `.caption2`, plus the gap that
    /// keeps two of them from reading as one word.
    private var labelWidth: CGFloat { periodType == .quarterly ? 62 : 44 }

    /// How many period labels the axis can carry without overprinting.
    var labelCapacity: Int {
        // Four until the first measurement lands: the count a phone can carry,
        // so the opening frame is never the overprinted one.
        guard plotWidth > 0 else { return 4 }
        return max(2, min(6, Int(plotWidth / labelWidth)))
    }

    /// Y ticks. Four on a narrow frame, where five would put labels closer
    /// together than the numbers they carry are wide.
    var yTickCount: Int { plotWidth > 0 && plotWidth < 300 ? 4 : 5 }

    /// Grouped bars stop reading as magnitudes once they are hairlines: past
    /// two dozen periods the question is the shape of the series, which is a
    /// line's job. Fifteen years of quarters is a line chart either way — and
    /// so is five years of quarters across four series on a phone, which is the
    /// same hairline arrived at from the other direction.
    var preferLines: Bool {
        if periodCount > StatementChartConfig.barToLineThreshold { return true }
        guard plotWidth > 0, periodCount > 0, seriesCount > 0 else { return false }
        return plotWidth / CGFloat(periodCount) / CGFloat(seriesCount) < 4
    }

    /// Every nth entry of `values`, anchored on the newest: the right-hand end
    /// is the one a reader looks for, and it must not be the label dropped.
    func thinned<T>(_ values: [T]) -> [T] {
        let capacity = labelCapacity
        guard values.count > capacity else { return values }
        let step = Int(ceil(Double(values.count) / Double(capacity)))
        return values.enumerated()
            .filter { (values.count - 1 - $0.offset) % step == 0 }
            .map(\.element)
    }
}

/// The frame a set of values should be plotted in. Swift Charts rounds its
/// automatic domain out to a whole tick past the extremes, so one small
/// quarterly loss in twenty buys a −20B gridline and hands half the frame to a
/// band no bar ever reaches — every real magnitude is squashed into what is
/// left to pay for it. Returns nil where there is no honest domain to pin and
/// Swift Charts' own is the better answer.
func periodChartDomain(_ values: [Double]) -> ClosedRange<Double>? {
    guard let lowest = values.min(), let highest = values.max() else { return nil }
    let low = min(0, lowest), high = max(0, highest)
    guard high > low else { return nil }
    let pad = (high - low) * 0.08
    return (low < 0 ? low - pad : 0)...(high > 0 ? high + pad : 0)
}

/// `chartYScale` only where the data supports one.
struct BoundedYScale: ViewModifier {
    let domain: ClosedRange<Double>?

    func body(content: Content) -> some View {
        if let domain {
            content.chartYScale(domain: domain)
        } else {
            content
        }
    }
}

/// One chart, one y-axis. Callers split series that cannot share a scale into
/// separate charts rather than reaching for a second axis.
struct StatementChartView: View {
    let periods: [String]
    let series: [StatementSeries]
    let periodType: StatementPeriod

    /// The width the card offers, which is what decides how many period labels
    /// the axis can carry and whether a bar is still wide enough to read. A
    /// phone gives this chart about a third of a desktop's frame, and the
    /// difference is the difference between six labels and a smear.
    @State private var chartWidth: CGFloat = 0

    private var metrics: PeriodChartMetrics {
        PeriodChartMetrics(
            containerWidth: chartWidth,
            periodCount: periods.count,
            seriesCount: series.count,
            periodType: periodType
        )
    }

    private var hasNegative: Bool { series.contains { $0.values.contains { ($0 ?? 0) < 0 } } }

    /// Only as many period labels as fit. A categorical x-scale draws a mark
    /// for every category it is given, so thinning has to happen here — twenty
    /// quarter labels overprint into a smear, which no font size fixes.
    private var axisPeriods: [String] { metrics.thinned(periods) }

    private var yDomain: ClosedRange<Double>? {
        periodChartDomain(series.flatMap { $0.values.compactMap { $0 } })
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
        VStack(alignment: .leading, spacing: 8) {
            legend
            chart
        }
        // Must be the *offered* width: measuring what the chart resolved to
        // would latch a wide label count on. See `readingContainerWidth`.
        .readingContainerWidth { chartWidth = $0 }
    }

    /// One series is named by its title; several need the legend, since
    /// identity must never rest on colour alone — which is also why this wraps
    /// rather than scrolls. A name you have to scroll to find is a name the
    /// reader never sees.
    private var legend: some View {
        WrappingRow(spacing: 14, lineSpacing: 6) {
            ForEach(series) { s in
                HStack(spacing: 5) {
                    RoundedRectangle(cornerRadius: 2).fill(s.color).frame(width: 9, height: 9)
                    Text(s.label).appFont(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
            }
        }
    }

    private var chart: some View {
        Chart {
            ForEach(series) { s in
                // Plotted against the period end itself, not its label: two
                // fiscal years can carry the same display year (Advance Auto
                // Parts closed years on 2022-01-01 and 2022-12-31) and a
                // categorical scale would merge them into one column.
                ForEach(Array(periods.enumerated()), id: \.offset) { i, period in
                    if let v = s.values.indices.contains(i) ? s.values[i] : nil {
                        // Wider than the default: the space between periods is
                        // worth less than a readable bar.
                        BarMark(
                            x: .value("Period", period),
                            y: .value("Value", v),
                            width: .ratio(0.9)
                        )
                            .foregroundStyle(s.color)
                            .position(by: .value("Series", s.label))
                            .cornerRadius(3)
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
        .modifier(BoundedYScale(domain: yDomain))
        .chartXAxis {
            AxisMarks(values: axisPeriods) { value in
                let iso = value.as(String.self)
                // A label centred on the outermost category hangs half its
                // width past the plot and is truncated there ("Jun 2…").
                // The edge labels hang inward instead.
                AxisValueLabel(anchor: edgeAnchor(iso)) {
                    if let iso {
                        Text(statementPeriodLabel(iso, periodType)).appFont(.caption2)
                    }
                }
            }
        }
        .chartYAxis {
            // Leading, as the web app draws it: the same statement should
            // not read right-to-left on one client and left-to-right on
            // another, and it frees the trailing edge for the newest
            // period's x label.
            AxisMarks(position: .leading, values: .automatic(desiredCount: metrics.yTickCount)) { value in
                AxisGridLine().foregroundStyle(.secondary.opacity(0.15))
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(formatStatementValue(v)).appFont(.caption2)
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
                        value: v.map(formatStatementValue) ?? "\u{2014}"
                    )
                }
            )
        }
    }
}

/// The statement picker and the period switch above the financials card.
///
/// These shared one row with a 190pt segmented control pinned to the trailing
/// edge, which on a phone left about 160pt for four statement chips that want
/// 430 — you saw "Income" and the left edge of "Balance", and the row read as
/// broken rather than as scrollable. Below the width where both fit, the switch
/// takes a row of its own and the chips get the whole card.
struct StatementTypeBar: View {
    let statement: String
    let period: StatementPeriod
    let onSelectStatement: (String) -> Void
    let onSelectPeriod: (StatementPeriod) -> Void

    @State private var width: CGFloat = 0

    private static let tabs = [
        ("income", "Income", "receipt"),
        ("balance", "Balance", "scalemass"),
        ("cash", "Cash Flow", "wallet.pass"),
        ("equity", "Equity", "person.2"),
    ]

    /// Four chips want about 430pt; below this the switch would be eating the
    /// room they need to be legible at all.
    private var stacked: Bool { prefersStackedLayout(measuredWidth: width, needs: 500) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if stacked {
                chips
                periodPicker.frame(maxWidth: .infinity)
            } else {
                HStack(alignment: .center, spacing: 12) {
                    chips
                    periodPicker.frame(width: 190)
                }
            }
        }
        // Must be the *offered* width. See `readingContainerWidth`.
        .readingContainerWidth { width = $0 }
    }

    /// Four chips want ~430pt and a phone card has ~330, so one of them is
    /// always partly out of frame. The selected one is brought fully into view
    /// — Equity is the last chip, and a selection you can see only half of
    /// reads as a rendering fault rather than as a row with more in it.
    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            ScrollViewReader { proxy in
                HStack(spacing: 8) {
                    ForEach(Self.tabs, id: \.0) { t in
                        Button { onSelectStatement(t.0) } label: {
                            HStack(spacing: 6) {
                                Image(systemName: t.2).appFont(.system(size: 16))
                                Text(t.1)
                            }
                            .appFont(.caption.weight(.bold))
                            .padding(.horizontal, 16).padding(.vertical, 8)
                            .foregroundStyle(statement == t.0 ? Color.white : .secondary)
                            .background(
                                statement == t.0 ? Color.indigo : Color.secondary.opacity(0.15),
                                in: Capsule()
                            )
                        }
                        .buttonStyle(.plain)
                        .id(t.0)
                    }
                }
                .onAppear { proxy.scrollTo(statement, anchor: .center) }
                .onChange(of: statement) { _, new in
                    withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(new, anchor: .center) }
                }
            }
        }
    }

    private var periodPicker: some View {
        Picker("", selection: Binding(get: { period }, set: onSelectPeriod)) {
            ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
    }
}

/// The card's own header: what is charted, over how many periods, and the
/// range switch. Stacks the switch under the title where a phone can't carry
/// both — the alternative is a segmented control squeezing "Quarterly trend"
/// into three wrapped lines.
struct StatementTrendHeader: View {
    let period: StatementPeriod
    let periodCount: Int
    @Binding var range: StatementRange

    @State private var width: CGFloat = 0

    /// Title, count and a 170pt switch need about this much to sit in one row.
    private var stacked: Bool { prefersStackedLayout(measuredWidth: width, needs: 380) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if stacked {
                title
                rangePicker.frame(maxWidth: .infinity)
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    title
                    Spacer(minLength: 12)
                    rangePicker.frame(width: 170)
                }
            }
        }
        // Must be the *offered* width. See `readingContainerWidth`.
        .readingContainerWidth { width = $0 }
    }

    /// Compresses rather than demands. `fixedSize()` here would give the wide
    /// branch a hard minimum of title + switch, which at a large Dynamic Type
    /// setting exceeds the threshold that chooses the branch — so the row would
    /// widen the page until the measurement agreed with it. See
    /// `prefersStackedLayout`.
    private var title: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text("\(period.title) trend")
                .appFont(.caption.weight(.bold)).textCase(.uppercase)
                .foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.7)
            Text("\(periodCount) \(period == .quarterly ? "quarters" : "years")")
                .appFont(.caption2).foregroundStyle(.tertiary)
                .lineLimit(1).minimumScaleFactor(0.7)
                .layoutPriority(-1)
        }
    }

    private var rangePicker: some View {
        Picker("", selection: $range) {
            ForEach(StatementRange.allCases) { r in Text(r.rawValue).tag(r) }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
    }
}

/// The three numbers a quarterly statement is usually opened for: where the
/// headline line item landed, how it moved on the prior period, and how it moved
/// on the same period a year earlier — the one comparison a seasonal business
/// can be judged on.
///
/// Three cells abreast is a desktop shape. Below the width where a title and a
/// figure both survive it, the headline takes a row of its own and the two
/// changes split the next — the numbers are the point, and a strip that renders
/// them as "VS PRIOR…" over a wrapped "+14.3\n%" has stopped carrying any.
struct StatementChangeStrip: View {
    let series: StatementSeries
    let periods: [String]
    let periodType: StatementPeriod

    @State private var width: CGFloat = 0

    private var lastIndex: Int? {
        series.values.lastIndex(where: { $0 != nil })
    }

    private var changeCount: Int { periodType == .quarterly ? 2 : 1 }

    /// A cell needs roughly this much to hold an uppercase title and a title3
    /// figure on one line each.
    private var stacked: Bool {
        prefersStackedLayout(measuredWidth: width, needs: CGFloat(changeCount + 1) * 152)
    }

    /// A percentage change off a negative or zero base says nothing.
    private func change(_ latest: Double, _ back: Int) -> Double? {
        guard let last = lastIndex, last - back >= 0,
              let prior = series.values[last - back], prior > 0 else { return nil }
        return (latest - prior) / prior * 100
    }

    var body: some View {
        if let last = lastIndex, let latest = series.values[last] {
            Group {
                if stacked {
                    VStack(spacing: 8) {
                        headline(latest, at: last)
                        HStack(alignment: .top, spacing: 8) { changes(latest) }
                    }
                } else {
                    HStack(alignment: .top, spacing: 12) {
                        headline(latest, at: last)
                        changes(latest)
                    }
                }
            }
            // Must be the *offered* width. See `readingContainerWidth`.
            .readingContainerWidth { width = $0 }
        }
    }

    /// The line item and its latest figure. The period end rides beside the
    /// number rather than inside the title: "Total Revenue · Jun 2026" is the
    /// string that truncated to "TOTAL…" and took the line item's name with it.
    private func headline(_ latest: Double, at index: Int) -> some View {
        cell(
            title: series.label,
            value: formatStatementValue(latest),
            footnote: periods.indices.contains(index)
                ? statementPeriodLabel(periods[index], periodType) : nil,
            tone: nil,
            dot: true
        )
    }

    @ViewBuilder private func changes(_ latest: Double) -> some View {
        if periodType == .quarterly {
            cell(title: "vs prior quarter", value: pct(change(latest, 1)), footnote: nil,
                 tone: change(latest, 1), dot: false)
        }
        let yearBack = periodType == .quarterly ? 4 : 1
        cell(
            title: periodType == .quarterly ? "vs year ago" : "vs prior year",
            value: pct(change(latest, yearBack)),
            footnote: nil,
            tone: change(latest, yearBack),
            dot: false
        )
    }

    private func pct(_ v: Double?) -> String {
        guard let v else { return "\u{2014}" }
        return String(format: "%@%.1f%%", v >= 0 ? "+" : "", v)
    }

    private func cell(title: String, value: String, footnote: String?, tone: Double?, dot: Bool) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                if dot { Circle().fill(series.color).frame(width: 7, height: 7) }
                Text(title).appFont(.caption2.weight(.bold)).textCase(.uppercase)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(value)
                    .appFont(.title3.bold().monospacedDigit())
                    .foregroundStyle(tone == nil ? Color.primary : (tone! >= 0 ? .green : .red))
                    // A figure never wraps: it shrinks, then it truncates.
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if let footnote {
                    Text(footnote).appFont(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
            }
        }
        // maxHeight so a two-line title in one cell doesn't leave the cells
        // beside it short — the backgrounds have to agree.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
    }
}
