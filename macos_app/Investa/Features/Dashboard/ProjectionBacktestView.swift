import SwiftUI
import Charts

/// The Backtest tab of `ProjectionCardView` (mirrors the web `ProjectionBacktest`):
/// the cone the model drew years ago with the path actually taken through it,
/// plus how often past outcomes really landed inside the bands.
/// Driven by `GET /projection/backtest`.
struct ProjectionBacktestView: View {
    let backtest: ProjectionBacktest?
    let currency: String
    let isLoading: Bool
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var isPhone: Bool { hSize == .compact }
    #else
    private var isPhone: Bool { false }
    #endif

    /// The path that actually happened.
    private static let actualColor = Color.green
    /// The model's own cone, in the same brand colour the forecast tab uses.
    private var modelColor: Color { Theme.brand }

    private var cur: String { backtest?.currency ?? currency }
    private var replay: ProjectionReplay? { backtest?.replay }
    private var horizons: [ProjectionBacktestHorizon] { backtest?.horizons ?? [] }
    private var points: [ProjectionReplayPoint] { replay?.points ?? [] }
    private var days: [Date] { points.compactMap(\.day) }

    /// An indexed replay (start = 100) is not money, so it must not be printed as money.
    private func money(_ value: Double) -> String {
        (replay?.indexed ?? false) ? Fmt.number(value, fractionDigits: 1) : Fmt.currency(value, code: cur)
    }

    private func compact(_ value: Double) -> String {
        (replay?.indexed ?? false) ? Fmt.number(value, fractionDigits: 0) : Fmt.compact(value, code: cur)
    }

    var body: some View {
        // Nil means "not fetched yet" (the tab was just opened, or a refresh
        // invalidated it); a failed fetch arrives as available == false instead.
        if backtest == nil || (isLoading && backtest?.available != true) {
            ProgressView().frame(maxWidth: .infinity).frame(height: 200)
        } else if let bt = backtest, bt.available {
            VStack(alignment: .leading, spacing: 12) {
                if replay != nil {
                    summary
                    outcomeLine
                    chart
                    legend
                }
                table(bt)
                footnote(bt)
            }
        } else {
            ContentUnavailableView(
                backtest?.reason == "error" ? "Backtest unavailable" : "Not enough history to backtest",
                systemImage: "clock.arrow.circlepath",
                description: Text(unavailableReason)
            )
            .frame(height: 200)
        }
    }

    private var unavailableReason: String {
        if backtest?.reason == "error" {
            return "Couldn't run the backtest just now. Refresh to try again."
        }
        guard let need = backtest?.requiredYears else {
            return "Backtesting appears once the portfolio has a longer track record."
        }
        let have = backtest?.historyYears ?? 0
        return String(
            format: "A backtest needs about %.0f years — enough to fit the model on the past and still have years left to check it against. This portfolio has %.1f.",
            need, have)
    }

    // MARK: - Summary

    private var summary: some View {
        let r = replay
        let anchor = r.map { "\(Int($0.years.rounded()))y from \(MarketTime.formatted($0.anchorDate))" } ?? ""
        return VStack(alignment: .leading, spacing: 4) {
            Text(anchor).appFont(.callout.weight(.bold)).lineLimit(1).minimumScaleFactor(0.7)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 16) { projectedStat; actualStat; bandStat }
                VStack(alignment: .leading, spacing: 4) { projectedStat; actualStat; bandStat }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var projectedStat: some View {
        stat("Projected", money(replay?.finalMedian ?? 0), color: .primary)
    }

    private var actualStat: some View {
        stat("Actual", money(replay?.finalActual ?? 0), color: Self.actualColor)
    }

    private var bandStat: some View {
        stat("10–90%", "\(compact(replay?.finalP10 ?? 0)) – \(compact(replay?.finalP90 ?? 0))", color: .secondary)
    }

    private func stat(_ label: String, _ value: String, color: Color) -> some View {
        HStack(spacing: 6) {
            Text(label).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
                .lineLimit(1)
            Text(value).appFont(.callout.weight(.bold)).foregroundStyle(color).lineLimit(1)
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private var outcomeLine: some View {
        let text: String
        switch replay?.outcome {
        case "below":
            text = "The actual path finished below the 10th percentile — the model was too optimistic over this stretch."
        case "above":
            text = "The actual path finished above the 90th percentile — the model was too cautious over this stretch."
        default:
            text = "The actual path finished inside the 10–90% band the model drew back then."
        }
        return Text(text)
            .appFont(.caption2).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Chart

    private var chart: some View {
        Chart {
            // Each band is its own series, otherwise Swift Charts interleaves the
            // two bands' points into one zigzag (sawtooth) path.
            ForEach(points) { p in
                if let d = p.day {
                    AreaMark(x: .value("Date", d), yStart: .value("Low", p.p10), yEnd: .value("High", p.p90))
                        .foregroundStyle(by: .value("Band", "10–90%"))
                        .interpolationMethod(.monotone)
                }
            }
            ForEach(points) { p in
                if let d = p.day {
                    AreaMark(x: .value("Date", d), yStart: .value("Low", p.p25), yEnd: .value("High", p.p75))
                        .foregroundStyle(by: .value("Band", "25–75%"))
                        .interpolationMethod(.monotone)
                }
            }
            ForEach(points) { p in
                if let d = p.day {
                    LineMark(x: .value("Date", d), y: .value("Projected", p.median))
                        .foregroundStyle(modelColor)
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 4]))
                        .interpolationMethod(.monotone)
                }
            }
            ForEach(points) { p in
                if let d = p.day, let actual = p.actual {
                    LineMark(x: .value("Date", d), y: .value("Actual", actual))
                        .foregroundStyle(Self.actualColor)
                        .lineStyle(StrokeStyle(lineWidth: 2.5))
                        .interpolationMethod(.monotone)
                }
            }
        }
        .chartForegroundStyleScale([
            "10–90%": modelColor.opacity(0.12),
            "25–75%": modelColor.opacity(0.22),
        ])
        .chartLegend(.hidden)
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: isPhone ? 3 : 5)) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let d = value.as(Date.self) { Text(MarketTime.monthYear(d)).fixedSize() }
                }
            }
        }
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) { Text(compact(v)).fixedSize() }
                }
            }
        }
        .chartHoverTooltip(days) { i in
            guard i < points.count else { return nil }
            let p = points[i]
            var rows = [ChartTooltipRow(color: modelColor, label: "Projected", value: money(p.median))]
            if let actual = p.actual {
                rows.append(ChartTooltipRow(color: Self.actualColor, label: "Actual", value: money(actual)))
            }
            rows.append(ChartTooltipRow(label: "10–90%", value: "\(compact(p.p10)) – \(compact(p.p90))"))
            return ChartTooltipContent(title: MarketTime.formatted(p.date), rows: rows)
        }
        .frame(height: 220)
    }

    private var legend: some View {
        WrappingRow(spacing: 14, lineSpacing: 4) {
            legendItem(Self.actualColor, "Actual (no later deposits)")
            legendItem(modelColor, "Projected median")
            legendItem(modelColor.opacity(0.25), "10–90% band")
        }
    }

    private func legendItem(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 10, height: 4)
            Text(label).appFont(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    // MARK: - Calibration table

    private func table(_ bt: ProjectionBacktest) -> some View {
        VStack(spacing: 0) {
            HStack {
                Text("Horizon").frame(maxWidth: .infinity, alignment: .leading)
                if !isPhone { Text("Checks").frame(maxWidth: .infinity, alignment: .trailing) }
                Text("Inside 10–90%").frame(maxWidth: .infinity, alignment: .trailing)
                if !isPhone { Text("Median actual").frame(maxWidth: .infinity, alignment: .trailing) }
                Text("Verdict").frame(maxWidth: .infinity, alignment: .trailing)
            }
            .appFont(.system(size: 11, weight: .bold)).textCase(.uppercase).tracking(0.5)
            .foregroundStyle(.secondary)
            .padding(.bottom, 6)

            ForEach(horizons) { h in
                Divider()
                HStack {
                    Text("\(h.years) \(h.years == 1 ? "year" : "years")")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if !isPhone {
                        Text("\(h.samples)")
                            .foregroundStyle(.secondary).monospacedDigit()
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    Text(Fmt.percent(h.inBandPct, fractionDigits: 0))
                        .fontWeight(.bold).monospacedDigit()
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    if !isPhone {
                        Text(Fmt.percent(h.medianActualReturnPct, includeSign: true))
                            .foregroundStyle(Self.actualColor).monospacedDigit()
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                    Text(Self.verdictLabel(h.verdict))
                        .fontWeight(.semibold)
                        .foregroundStyle(Self.verdictColor(h.verdict))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .appFont(.subheadline)
                .lineLimit(1).minimumScaleFactor(0.6)
                .padding(.vertical, 8)
            }
        }
    }

    private static func verdictLabel(_ verdict: String) -> String {
        switch verdict {
        case "narrow": return "Too narrow"
        case "wide": return "Conservative"
        default: return "Calibrated"
        }
    }

    private static func verdictColor(_ verdict: String) -> Color {
        switch verdict {
        case "narrow": return .red
        case "wide": return .secondary
        default: return .green
        }
    }

    private func footnote(_ bt: ProjectionBacktest) -> some View {
        var window = ""
        if let start = bt.historyStart, let end = bt.historyEnd {
            window = " (\(MarketTime.formatted(start)) – \(MarketTime.formatted(end)))"
        }
        return Text("Walk-forward test on this portfolio's own history\(window): at each month in the past the model is refitted on the data that existed then — never later — and its cone scored against what followed. \"Inside 10–90%\" should come out near 80%: much less and the bands are too narrow to trust, much more and they are wider than they need to be. Returns are time-weighted, so deposits and withdrawals after each start date don't flatter the result.")
            .appFont(.caption2).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
