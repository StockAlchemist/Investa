import SwiftUI
import Charts

/// The valuation / earnings / profitability / market readings for one company,
/// presented as either a dense 4-panel grid or historical time-series graphs.
///
/// Mirrors `web_app/components/StockKeyMetrics.tsx`.
struct StockKeyMetricsView: View {
    let metrics: [String: Double]
    var beta: Double?
    var averageVolume: Double?
    var viewModel: StockDetailViewModel?

    @Environment(\.horizontalSizeClass) private var hSizeClass
    @State private var viewMode: ViewMode = .grid
    @State private var selectedGroup: String = "Valuation"

    enum ViewMode: String, CaseIterable {
        case grid = "Table"
        case graphs = "Graphs"
    }

    struct MetricChartDef: Identifiable {
        var id: String { dataKey }
        let group: String
        let dataKey: String
        let title: String
        let color: Color
        var isPercent: Bool = false
        var isCount: Bool = false
    }

    static let chartDefs: [MetricChartDef] = [
        // 1. Valuation
        MetricChartDef(group: "Valuation", dataKey: "P/E Ratio", title: "Price to Earnings (P/E)", color: Color(hex: 0x10b981)),
        MetricChartDef(group: "Valuation", dataKey: "P/S Ratio", title: "Price to Sales (P/S)", color: Color(hex: 0x06b6d4)),
        MetricChartDef(group: "Valuation", dataKey: "P/B Ratio", title: "Price to Book (P/B)", color: Color(hex: 0x8b5cf6)),
        MetricChartDef(group: "Valuation", dataKey: "EV/EBITDA", title: "EV / EBITDA", color: Color(hex: 0xf59e0b)),
        MetricChartDef(group: "Valuation", dataKey: "EV/Sales", title: "EV / Sales", color: Color(hex: 0xec4899)),
        MetricChartDef(group: "Valuation", dataKey: "P/FCF Ratio", title: "Price to Free Cash Flow (P/FCF)", color: Color(hex: 0x14b8a6)),
        MetricChartDef(group: "Valuation", dataKey: "Dividend Yield (%)", title: "Dividend Yield", color: Color(hex: 0x10b981), isPercent: true),

        // 2. Profitability
        MetricChartDef(group: "Profitability", dataKey: "Return on Invested Capital (ROIC) (%)", title: "Return on Invested Capital (ROIC)", color: Color(hex: 0xec4899), isPercent: true),
        MetricChartDef(group: "Profitability", dataKey: "Return on Equity (ROE) (%)", title: "Return on Equity (ROE)", color: Color(hex: 0x10b981), isPercent: true),
        MetricChartDef(group: "Profitability", dataKey: "Return on Assets (ROA) (%)", title: "Return on Assets (ROA)", color: Color(hex: 0x06b6d4), isPercent: true),
        MetricChartDef(group: "Profitability", dataKey: "Gross Profit Margin (%)", title: "Gross Margin", color: Color(hex: 0x8b5cf6), isPercent: true),
        MetricChartDef(group: "Profitability", dataKey: "Net Profit Margin (%)", title: "Net Margin", color: Color(hex: 0xf59e0b), isPercent: true),
        MetricChartDef(group: "Profitability", dataKey: "Free Cash Flow Margin (%)", title: "Free Cash Flow Margin", color: Color(hex: 0x14b8a6), isPercent: true),

        // 3. Balance Sheet
        MetricChartDef(group: "Balance Sheet", dataKey: "Current Ratio", title: "Current Ratio", color: Color(hex: 0x10b981)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Quick Ratio", title: "Quick Ratio", color: Color(hex: 0x06b6d4)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Debt-to-Equity Ratio", title: "Debt to Equity", color: Color(hex: 0xf59e0b)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Long-Term Debt to Equity", title: "LT Debt to Equity", color: Color(hex: 0x8b5cf6)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Interest Coverage Ratio", title: "Interest Coverage Ratio", color: Color(hex: 0xec4899)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Asset Turnover", title: "Asset Turnover", color: Color(hex: 0x06b6d4)),
        MetricChartDef(group: "Balance Sheet", dataKey: "Diluted Shares Outstanding", title: "Diluted Shares Outstanding", color: Color(hex: 0x64748b), isCount: true),

        // 4. Earnings & Sales
        MetricChartDef(group: "Earnings & Sales", dataKey: "Diluted EPS", title: "Diluted EPS ($)", color: Color(hex: 0x10b981)),
        MetricChartDef(group: "Earnings & Sales", dataKey: "Total Revenue", title: "Total Revenue (Sales)", color: Color(hex: 0x06b6d4), isCount: true),
        MetricChartDef(group: "Earnings & Sales", dataKey: "Revenue Growth YoY (%)", title: "Revenue Growth YoY", color: Color(hex: 0x8b5cf6), isPercent: true),
        MetricChartDef(group: "Earnings & Sales", dataKey: "EPS Growth YoY (%)", title: "EPS Growth YoY", color: Color(hex: 0xec4899), isPercent: true),
        MetricChartDef(group: "Earnings & Sales", dataKey: "Operating Margin (%)", title: "Operating Margin", color: Color(hex: 0xf59e0b), isPercent: true),
    ]

    private var columns: Int { hSizeClass == .regular ? 2 : 1 }

    private var populatedGroups: [StockMetric.Group] {
        StockMetric.Group.allCases.filter { group in
            StockMetric.inGroup(group).contains { metrics[$0.field] != nil }
                || (group == .market && !extraRows.isEmpty)
        }
    }

    private var extraRows: [(label: String, value: String)] {
        var rows: [(String, String)] = []
        if let beta { rows.append(("Beta", String(format: "%.2f", beta))) }
        if let averageVolume { rows.append(("Avg Volume", StockMetric.compactCount(averageVolume))) }
        return rows
    }

    var body: some View {
        if !populatedGroups.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                headerRow

                if viewMode == .grid {
                    LazyVGrid(
                        columns: Array(repeating: GridItem(.flexible(), spacing: 12, alignment: .top),
                                       count: columns),
                        alignment: .leading, spacing: 12
                    ) {
                        ForEach(populatedGroups, id: \.self) { group in
                            panel(group)
                        }
                    }
                } else if let vm = viewModel {
                    graphsView(vm)
                }
            }
            .onChange(of: viewMode) { _, mode in
                if mode == .graphs, let vm = viewModel, vm.ratios == nil {
                    Task { await vm.loadRatios() }
                }
            }
        }
    }

    // MARK: - Header

    private var headerRow: some View {
        HStack(alignment: .center) {
            HStack(spacing: 8) {
                Label("Key Metrics", systemImage: "square.grid.3x3").font(.headline)

                if viewModel != nil {
                    Picker("", selection: $viewMode) {
                        ForEach(ViewMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 140)
                }
            }

            Spacer(minLength: 12)

            if viewMode == .grid {
                legend
            } else if let vm = viewModel {
                periodPicker(vm)
            }
        }
    }

    private var legend: some View {
        HStack(spacing: 4) {
            Text("Green").foregroundStyle(Color.up).fontWeight(.semibold)
            Text("beats,")
            Text("red").foregroundStyle(Color.down).fontWeight(.semibold)
            Text("trails a typical S&P 500 company")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
    }

    private func periodPicker(_ vm: StockDetailViewModel) -> some View {
        Picker("", selection: Binding(
            get: { vm.ratiosPeriod },
            set: { p in Task { await vm.loadRatios(period: p) } }
        )) {
            ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
        }
        .pickerStyle(.segmented)
        .frame(width: 160)
    }

    // MARK: - Grid Panel View

    private func panel(_ group: StockMetric.Group) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(group.rawValue)
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .padding(.bottom, 6)

            ForEach(StockMetric.inGroup(group)) { metric in
                let value = metrics[metric.field]
                row(
                    metric.label,
                    value.map { metric.formatted($0) } ?? "–",
                    value.map { metric.tone($0) } ?? .secondary.opacity(0.5)
                ) {
                    if viewModel != nil {
                        let grpName: String = (group == .market ? "Valuation" : group.rawValue)
                        selectedGroup = grpName
                        viewMode = .graphs
                    }
                }
                Divider().opacity(0.4)
            }
            if group == .market {
                ForEach(extraRows, id: \.label) { extra in
                    row(extra.label, extra.value, .primary)
                    Divider().opacity(0.4)
                }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func row(_ label: String, _ value: String, _ tone: Color, action: (() -> Void)? = nil) -> some View {
        Button {
            action?()
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 8)
                Text(value)
                    .font(.caption.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(tone)
                    .lineLimit(1)
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Graphs View

    @ViewBuilder
    private func graphsView(_ vm: StockDetailViewModel) -> some View {
        let groups = ["Valuation", "Earnings & Sales", "Profitability", "Balance Sheet"]
        let history = vm.ratios?.historical ?? []
        let activeDefs = Self.chartDefs.filter { $0.group == selectedGroup }

        VStack(alignment: .leading, spacing: 14) {
            // Category selector
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(groups, id: \.self) { grp in
                        Button {
                            selectedGroup = grp
                        } label: {
                            Text(grp)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 10).padding(.vertical, 5)
                                .background(selectedGroup == grp ? Color.accentColor : Color.gray.opacity(0.12), in: Capsule())
                                .foregroundStyle(selectedGroup == grp ? Color.white : Color.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            if vm.isLoadingRatios {
                ProgressView().frame(maxWidth: .infinity, minHeight: 180)
            } else if history.isEmpty {
                ContentUnavailableView("No ratio data available", systemImage: "chart.line.uptrend.xyaxis")
                    .frame(maxWidth: .infinity, minHeight: 180)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 280), spacing: 14)], spacing: 14) {
                    ForEach(activeDefs) { def in
                        chartCard(def, history: history, period: vm.ratiosPeriod)
                    }
                }
            }
        }
    }

    private func chartCard(_ def: MetricChartDef, history: [[String: JSONValue]], period: StatementPeriod) -> some View {
        let points: [(period: Date, iso: String, value: Double)] = history.reversed().compactMap { item in
            guard let val = item[def.dataKey]?.doubleValue,
                  let dateStr = item["Period"]?.stringValue,
                  let period = MarketTime.calendarDay(dateStr) else { return nil }
            return (period, dateStr, val)
        }

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(def.title)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .lineLimit(1)
                Spacer()
                if let last = points.last {
                    Text(formatRatioVal(last.value, isPercent: def.isPercent, isCount: def.isCount))
                        .font(.caption.weight(.bold))
                        .monospacedDigit()
                        .foregroundStyle(def.color)
                }
            }

            if points.isEmpty {
                Text("No data filed").font(.caption2).foregroundStyle(.secondary).frame(height: 140)
            } else {
                Chart {
                    ForEach(Array(points.enumerated()), id: \.offset) { _, p in
                        AreaMark(x: .value("Period", p.period), y: .value(def.title, p.value))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [def.color.opacity(0.30), def.color.opacity(0.02)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                            .interpolationMethod(.monotone)

                        LineMark(x: .value("Period", p.period), y: .value(def.title, p.value))
                            .foregroundStyle(def.color)
                            .lineStyle(.init(lineWidth: 2.0))
                            .interpolationMethod(.monotone)
                    }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 4)) { value in
                        AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                        if let date = value.as(Date.self) {
                            AxisValueLabel {
                                Text(period == .quarterly ? MarketTime.monthYear(date) : MarketTime.year(date))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                        AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                        if let v = value.as(Double.self) {
                            AxisValueLabel {
                                Text(formatRatioVal(v, isPercent: def.isPercent, isCount: def.isCount))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .frame(height: 140)
                .chartHoverTooltip(points.map(\.period)) { i in
                    guard points.indices.contains(i) else { return nil }
                    let p = points[i]
                    return ChartTooltipContent(
                        title: MarketTime.formatted(p.iso),
                        rows: [
                            ChartTooltipRow(
                                color: def.color,
                                label: def.title,
                                value: formatRatioVal(p.value, isPercent: def.isPercent, isCount: def.isCount)
                            )
                        ]
                    )
                }
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func formatRatioVal(_ v: Double, isPercent: Bool, isCount: Bool) -> String {
        if isCount { return StockMetric.compactCount(v) }
        return isPercent ? Fmt.percent(v) : Fmt.number(v, fractionDigits: 2)
    }
}
