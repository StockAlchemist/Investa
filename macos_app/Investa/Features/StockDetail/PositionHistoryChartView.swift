import SwiftUI
import Charts

struct PositionHistoryChartView: View {
    let symbol: String
    let currency: String
    @ObservedObject var viewModel: StockDetailViewModel

    @State private var viewMode: ViewMode = .value
    @State private var period: String = "1y"
    @State private var selectedBenchmarks: [String] = []

    enum ViewMode: String, CaseIterable {
        case value = "Value"
        case returnPct = "Return %"
    }

    private struct Benchmark {
        let name: String
        let color: Color
    }

    private let benchmarks = [
        Benchmark(name: "S&P 500", color: Color(hex: 0xf59e0b)),
        Benchmark(name: "NASDAQ", color: Color(hex: 0x8b5cf6)),
        Benchmark(name: "Dow Jones", color: Color(hex: 0x0ea5e9)),
    ]

    private let periods: [(String, String)] = [
        ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("YTD", "ytd"),
        ("1Y", "1y"), ("3Y", "3y"), ("5Y", "5y"), ("ALL", "all"),
    ]

    private var pts: [StockPositionHistoryPoint] {
        viewModel.positionHistory
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header: Title + Badge + Subtitle + Controls
            headerRow

            // Benchmark Toggles (in Return mode)
            if viewMode == .returnPct {
                benchmarkRow
            }

            // Chart Area
            chartArea
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
        .task {
            await reloadHistory()
        }
        .onChange(of: period) { _, _ in
            Task { await reloadHistory() }
        }
        .onChange(of: selectedBenchmarks) { _, _ in
            Task { await reloadHistory() }
        }
    }

    private func reloadHistory() async {
        await viewModel.loadPositionHistory(period: period, benchmarks: selectedBenchmarks)
    }

    // MARK: - Header

    @ViewBuilder
    private var headerRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: viewMode == .value ? "chart.pie.fill" : "chart.line.uptrend.xyaxis")
                        .foregroundStyle(viewMode == .value ? Color.indigo : Color.green)
                        .font(.system(size: 14))
                    Text("Position History")
                        .font(.headline)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }

                if let last = pts.last {
                    if last.shares < 1e-6 {
                        if viewMode == .value {
                            Text("Closed Position")
                                .font(.caption2.weight(.bold))
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12), in: Capsule())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                        } else {
                            let isPositive = last.returnPct >= 0
                            Text("\(last.returnPct >= 0 ? "+" : "")\(Fmt.percent(last.returnPct))")
                                .font(.caption2.weight(.bold))
                                .padding(.horizontal, 6).padding(.vertical, 2)
                                .background(isPositive ? Color.green.opacity(0.12) : Color.red.opacity(0.12), in: Capsule())
                                .foregroundStyle(isPositive ? Color.green : Color.red)
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                    } else {
                        let isPositive = viewMode == .value ? (last.unrealizedGain >= 0) : (last.returnPct >= 0)
                        Text(viewMode == .value
                             ? "\(last.unrealizedGain >= 0 ? "+" : "")\(Fmt.currency(last.unrealizedGain, currency: currency)) (\(Fmt.percent(last.unrealizedGainPct)))"
                             : "\(last.returnPct >= 0 ? "+" : "")\(Fmt.percent(last.returnPct))")
                            .font(.caption2.weight(.bold))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(isPositive ? Color.green.opacity(0.12) : Color.red.opacity(0.12), in: Capsule())
                            .foregroundStyle(isPositive ? Color.green : Color.red)
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                }

                Spacer()
            }

            Text(viewMode == .value
                 ? "Market value & cost basis over time\(pts.last?.shares ?? 0 < 1e-6 ? " (Position closed)" : "")"
                 : "Holding return % over selected period")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            // Controls Row: View Switcher and Period Selector
            HStack(spacing: 8) {
                Picker("", selection: $viewMode) {
                    ForEach(ViewMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 155)

                periodRow
            }
            .padding(.top, 2)
        }
    }

    // MARK: - Controls

    private var periodRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(periods, id: \.1) { label, value in
                    Button {
                        period = value
                    } label: {
                        Text(label)
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(period == value ? Color.accentColor : Color.gray.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                            .foregroundStyle(period == value ? .white : .secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var benchmarkRow: some View {
        HStack(spacing: 8) {
            Text("COMPARE")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(benchmarks, id: \.name) { b in
                        let isSelected = selectedBenchmarks.contains(b.name)
                        Button {
                            if isSelected {
                                selectedBenchmarks.removeAll { $0 == b.name }
                            } else {
                                selectedBenchmarks.append(b.name)
                            }
                        } label: {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(isSelected ? .white : b.color)
                                    .frame(width: 6, height: 6)
                                Text(b.name)
                                    .font(.system(size: 10, weight: .bold))
                            }
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(isSelected ? b.color : Color.gray.opacity(0.12), in: Capsule())
                            .foregroundStyle(isSelected ? .white : .secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - Chart

    private var yDomain: ClosedRange<Double> {
        guard !pts.isEmpty else { return 0...100 }
        if viewMode == .value {
            let maxVal = pts.map { max($0.value, $0.costBasis) }.max() ?? 100
            return 0...(max(1, maxVal * 1.08))
        } else {
            var allVals = pts.map(\.returnPct)
            for b in selectedBenchmarks {
                allVals += pts.compactMap { $0.benchmarks[b] }
            }
            let minVal = min(0, allVals.min() ?? 0)
            let maxVal = max(0, allVals.max() ?? 0)
            let pad = max(5.0, (maxVal - minVal) * 0.1)
            return (minVal - pad)...(maxVal + pad)
        }
    }

    @ViewBuilder
    private var chartArea: some View {
        if viewModel.isLoadingPositionHistory && pts.isEmpty {
            ProgressView()
                .frame(maxWidth: .infinity, minHeight: 240)
        } else if pts.isEmpty {
            ContentUnavailableView("No history data", systemImage: "chart.xyaxis.line")
                .frame(maxWidth: .infinity, minHeight: 240)
        } else {
            chart
                .frame(height: 250)
        }
    }

    private func formatXDate(_ d: Date) -> String {
        let f = DateFormatter()
        f.timeZone = TimeZone(identifier: "America/New_York")
        if ["1m", "3m", "6m"].contains(period.lowercased()) {
            f.dateFormat = "MMM d"
        } else {
            f.dateFormat = "MMM ''yy"
        }
        return f.string(from: d)
    }

    private var chart: some View {
        return Chart {
            if viewMode == .value {
                // 1. Market Value Area Fill
                ForEach(pts) { p in
                    if let d = p.parsedDate {
                        AreaMark(
                            x: .value("Date", d),
                            yStart: .value("Floor", 0),
                            yEnd: .value("Value", p.value)
                        )
                        .foregroundStyle(
                            .linearGradient(
                                colors: [Color.indigo.opacity(0.30), Color.indigo.opacity(0.02)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .interpolationMethod(.monotone)
                    }
                }

                // 2. Market Value Continuous Line
                ForEach(pts) { p in
                    if let d = p.parsedDate {
                        LineMark(
                            x: .value("Date", d),
                            y: .value("Market Value", p.value),
                            series: .value("Series", "Market Value")
                        )
                        .foregroundStyle(Color.indigo)
                        .lineStyle(.init(lineWidth: 2.0))
                        .interpolationMethod(.monotone)
                    }
                }

                // 3. Cost Basis Line (Dashed)
                if pts.contains(where: { $0.costBasis > 0 }) {
                    ForEach(pts) { p in
                        if let d = p.parsedDate {
                            LineMark(
                                x: .value("Date", d),
                                y: .value("Cost Basis", p.costBasis),
                                series: .value("Series", "Cost Basis")
                            )
                            .foregroundStyle(Color(hex: 0x94a3b8))
                            .lineStyle(.init(lineWidth: 1.5, dash: [4, 4]))
                            .interpolationMethod(.monotone)
                        }
                    }
                }
            } else {
                // Return % View
                let isPos = (pts.last?.returnPct ?? 0) >= 0
                let strokeColor = isPos ? Color.green : Color.red

                RuleMark(y: .value("Zero", 0))
                    .foregroundStyle(Color.secondary.opacity(0.35))
                    .lineStyle(.init(lineWidth: 1, dash: [3, 3]))

                // 1. Return % Area Fill
                ForEach(pts) { p in
                    if let d = p.parsedDate {
                        AreaMark(
                            x: .value("Date", d),
                            yStart: .value("Floor", 0),
                            yEnd: .value("Return", p.returnPct)
                        )
                        .foregroundStyle(
                            .linearGradient(
                                colors: [strokeColor.opacity(0.28), strokeColor.opacity(0.02)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .interpolationMethod(.monotone)
                    }
                }

                // 2. Return % Continuous Line
                ForEach(pts) { p in
                    if let d = p.parsedDate {
                        LineMark(
                            x: .value("Date", d),
                            y: .value("Position Return", p.returnPct),
                            series: .value("Series", "Position Return")
                        )
                        .foregroundStyle(strokeColor)
                        .lineStyle(.init(lineWidth: 2.0))
                        .interpolationMethod(.monotone)
                    }
                }

                // 3. Benchmark overlays
                ForEach(selectedBenchmarks, id: \.self) { bmName in
                    let bmColor = benchmarks.first(where: { $0.name == bmName })?.color ?? .orange
                    ForEach(pts) { p in
                        if let d = p.parsedDate, let bVal = p.benchmarks[bmName] {
                            LineMark(
                                x: .value("Date", d),
                                y: .value(bmName, bVal),
                                series: .value("Series", bmName)
                            )
                            .foregroundStyle(bmColor)
                            .lineStyle(.init(lineWidth: 1.5))
                            .interpolationMethod(.monotone)
                        }
                    }
                }
            }
        }
        .chartYScale(domain: yDomain)
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { v in
                AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                AxisValueLabel {
                    if let d = v.as(Double.self) {
                        Text(viewMode == .value
                             ? Fmt.compact(d, code: currency)
                             : "\(d >= 0 ? "+" : "")\(Fmt.number(d, fractionDigits: 0))%")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisGridLine().foregroundStyle(Color.secondary.opacity(0.15))
                if let date = value.as(Date.self) {
                    AxisValueLabel {
                        Text(formatXDate(date))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .chartHoverTooltip(pts.compactMap(\.parsedDate)) { i in
            tooltip(pts[i])
        }
    }

    private func tooltip(_ p: StockPositionHistoryPoint) -> ChartTooltipContent {
        var rows: [ChartTooltipRow] = []

        if viewMode == .value {
            rows.append(ChartTooltipRow(color: .indigo, label: "Market Value", value: Fmt.currency(p.value, currency: currency)))
            rows.append(ChartTooltipRow(color: Color(hex: 0x94a3b8), label: "Cost Basis", value: Fmt.currency(p.costBasis, currency: currency)))
            if p.costBasis > 0 {
                rows.append(ChartTooltipRow(
                    color: p.unrealizedGain >= 0 ? .green : .red,
                    label: "Unrealized G/L",
                    value: "\(p.unrealizedGain >= 0 ? "+" : "")\(Fmt.currency(p.unrealizedGain, currency: currency)) (\(Fmt.percent(p.unrealizedGainPct)))"
                ))
            }
            if p.shares > 0 {
                rows.append(ChartTooltipRow(color: .secondary, label: "Shares", value: "\(Fmt.number(p.shares, fractionDigits: 4)) sh"))
            }
        } else {
            let isPos = p.returnPct >= 0
            rows.append(ChartTooltipRow(
                color: isPos ? .green : .red,
                label: "Position Return",
                value: "\(isPos ? "+" : "")\(Fmt.percent(p.returnPct))"
            ))
            for bmName in selectedBenchmarks {
                if let bVal = p.benchmarks[bmName] {
                    let bmColor = benchmarks.first(where: { $0.name == bmName })?.color ?? .orange
                    rows.append(ChartTooltipRow(
                        color: bmColor,
                        label: bmName,
                        value: "\(bVal >= 0 ? "+" : "")\(Fmt.percent(bVal))"
                    ))
                }
            }
        }

        return ChartTooltipContent(title: p.date, rows: rows)
    }
}
