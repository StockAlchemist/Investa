import SwiftUI

/// The market-trend indicator, as a card.
///
/// **Advisory only.** No strategy acts on this signal — gating the stock book
/// with it was measured and rejected (13.0%/yr against 16.3% for staying
/// invested, with a deeper drawdown). The card therefore describes the market
/// and never instructs; a card that reads like an instruction is one people
/// will follow.
///
/// It still keeps two readings apart. The headline is the **active** one, set
/// at the last completed month-end. The provisional reading appears only as a
/// secondary note when it diverges, so a mid-month price is never mistaken for
/// the current state.
struct TrendSignalCard: View {
    let signal: TrendSignal
    var compact = false

    private var isUp: Bool { signal.state == .up }

    private var accent: Color { isUp ? .up : .orange }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if !compact {
                stats
                footnote
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Market trend · \(signal.signalSymbol) \(signal.smaMonths)-month")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                HStack(spacing: 7) {
                    Image(systemName: isUp ? "arrow.up.right.circle.fill" : "arrow.down.right.circle.fill")
                        .foregroundStyle(accent)
                    Text(signal.state.label)
                        .font(.title3.weight(.bold))
                        .foregroundStyle(accent)
                }

                Text((isUp ? "Trading above its moving average."
                           : "Trading below its moving average.")
                     + " Context only — no strategy acts on this.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)

            if !compact {
                SignalSparkline(points: signal.history)
                    .frame(width: 92, height: 30)
            }
        }
    }

    private var stats: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
            GridRow {
                stat("Set at", "\(Self.day(signal.decisionDate)) · \(Self.money(signal.decisionClose))")
                stat("\(signal.smaMonths)-month average", Self.money(signal.sma))
            }
            GridRow {
                stat("Governs", Self.month(signal.governsMonth))
                stat("Next check", Self.day(signal.nextDecisionDate))
            }
        }
    }

    @ViewBuilder
    private var footnote: some View {
        if signal.wouldFlip {
            let target = signal.provisionalState == .up ? "up" : "down"
            Text("**On track to turn \(target)** at the \(Self.day(signal.nextDecisionDate)) close "
                 + "— the reading only changes on month-end closes.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.12)))
        } else {
            let direction = isUp ? "below" : "above"
            let distance = signal.distancePct.map { String(format: " — %.1f%% away.", abs($0)) } ?? "."
            Text("Unchanged unless \(signal.signalSymbol) closes \(direction) \(Self.money(signal.flipClose)) "
                 + "on \(Self.day(signal.nextDecisionDate))\(distance)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption.weight(.medium)).monospacedDigit()
        }
    }

    // MARK: - Formatting

    private static func money(_ value: Double) -> String {
        String(format: "$%.2f", value)
    }

    /// `2026-06-30` -> `30 Jun`. Falls back to the raw string rather than
    /// showing a wrong date if the backend ever changes the format.
    private static func day(_ iso: String) -> String {
        let parser = DateFormatter()
        parser.dateFormat = "yyyy-MM-dd"
        parser.timeZone = TimeZone(identifier: "UTC")
        guard let date = parser.date(from: iso) else { return iso }
        let out = DateFormatter()
        out.dateFormat = "d MMM"
        out.timeZone = TimeZone(identifier: "UTC")
        return out.string(from: date)
    }

    /// `2026-07` -> `July 2026`.
    private static func month(_ value: String) -> String {
        let parser = DateFormatter()
        parser.dateFormat = "yyyy-MM"
        parser.timeZone = TimeZone(identifier: "UTC")
        guard let date = parser.date(from: value) else { return value }
        let out = DateFormatter()
        out.dateFormat = "MMMM yyyy"
        out.timeZone = TimeZone(identifier: "UTC")
        return out.string(from: date)
    }
}

/// Month-end closes against their moving average. No axes — the shape is the
/// message, and the numbers are already in the stat grid beside it.
private struct SignalSparkline: View {
    let points: [TrendSignal.Point]

    /// Only points with an average to compare against — the first months of the
    /// series have no moving average yet and would draw a line to nowhere.
    private var usable: [TrendSignal.Point] { points.filter { $0.sma != nil } }

    var body: some View {
        GeometryReader { geo in
            if usable.count >= 2 {
                path(in: geo.size) { $0.sma ?? $0.close }
                    .stroke(Color.secondary.opacity(0.45),
                            style: StrokeStyle(lineWidth: 1.5, dash: [3, 2]))
                path(in: geo.size) { $0.close }
                    .stroke(Color.primary.opacity(0.7), lineWidth: 1.5)
            }
        }
    }

    /// Both series share one scale, so the crossings the rule acts on stay
    /// visible; scaling them independently would draw a chart where the line
    /// crosses its own average at the wrong place.
    private func path(in size: CGSize, pick: (TrendSignal.Point) -> Double) -> Path {
        let series = usable
        let values = series.flatMap { [$0.close, $0.sma ?? $0.close] }
        let low = values.min() ?? 0
        let high = values.max() ?? 1
        let span = max(high - low, 0.0001)

        return Path { p in
            for (index, point) in series.enumerated() {
                let x = size.width * CGFloat(index) / CGFloat(series.count - 1)
                let y = size.height * (1 - CGFloat((pick(point) - low) / span))
                if index == 0 {
                    p.move(to: CGPoint(x: x, y: y))
                } else {
                    p.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }
    }
}

/// Self-loading wrapper for placements that have no allocation to draw from.
///
/// The dashboard shows the indicator on its own, so it fetches its own data
/// rather than borrowing a strategy's — which is the honest arrangement now
/// that no strategy carries this signal.
struct TrendSignalCardLoader: View {
    var symbol: String = "QQQ"
    var smaMonths: Int = 10

    @State private var signal: TrendSignal?
    @State private var failed = false

    var body: some View {
        Group {
            if let signal {
                TrendSignalCard(signal: signal)
            } else if failed {
                Label("Market trend unavailable", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .card()
            } else {
                ProgressView()
                    .padding(14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .card()
            }
        }
        .task {
            do {
                signal = try await APIClient.shared.get(
                    "/trend-signal",
                    query: [
                        URLQueryItem(name: "symbol", value: symbol),
                        URLQueryItem(name: "sma_months", value: String(smaMonths)),
                    ]
                )
            } catch {
                failed = true
            }
        }
    }
}
