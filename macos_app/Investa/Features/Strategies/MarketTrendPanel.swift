import SwiftUI

/// One market the trend panel reads.
///
/// `all` mirrors `MARKET_SIGNAL_INDICES` in `src/strategies.py` (and
/// `MARKET_TREND_INDICES` in `web_app/lib/api.ts`). Both legs are ETFs rather
/// than the raw indices so the two moving averages are built from the same kind
/// of series — a crossing in one is then comparable to a crossing in the other.
/// The label here is only a fallback: the backend names each reading.
struct MarketTrendIndex: Identifiable, Sendable {
    let symbol: String
    let label: String

    var id: String { symbol }

    static let all: [MarketTrendIndex] = [
        MarketTrendIndex(symbol: "SPY", label: "S&P 500"),
        MarketTrendIndex(symbol: "QQQ", label: "NASDAQ 100"),
    ]
}

/// The market-trend panel: one moving-average reading per market, stacked.
///
/// **Advisory only.** No strategy acts on these signals — gating the stock book
/// with the NASDAQ reading was measured and rejected (13.0%/yr against 16.3%
/// for staying invested, with a deeper drawdown). The panel therefore describes
/// the markets and never instructs; a panel that reads like an instruction is
/// one people will follow.
///
/// **Why more than one market.** A single index reads as a verdict on "the
/// market". Two that disagree as often as the S&P 500 and the NASDAQ 100 show
/// how much of the answer depends on which index was picked — the honest way to
/// present an indicator nothing acts on. Each reading is fetched independently,
/// so one market's price feed failing still leaves the other readable.
///
/// Each row keeps two readings apart: the headline state is the **active** one,
/// set at the last completed month-end, while the provisional reading appears
/// only as a note when it diverges, so a mid-month price is never mistaken for
/// the current state. The timing the rows share — the month they govern, the
/// next check — is stated once in the footer instead of per row.
struct MarketTrendPanel: View {
    var indices: [MarketTrendIndex] = MarketTrendIndex.all
    var smaMonths: Int = 10

    @State private var signals: [String: TrendSignal] = [:]
    @State private var loaded = false

    /// Readings in display order — the source for the shared footer.
    private var available: [TrendSignal] { indices.compactMap { signals[$0.symbol] } }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header

            if !loaded {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 10)
            } else if available.isEmpty {
                Label("Market trend unavailable", systemImage: "exclamationmark.triangle")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 6)
            } else {
                rows
                footnote
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
        .task { await load() }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Market trend")
                .appFont(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Spacer(minLength: 8)
            Text("\(smaMonths)-month average")
                .appFont(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
        }
    }

    private var rows: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(indices.enumerated()), id: \.element.id) { offset, index in
                if offset > 0 {
                    Divider().opacity(0.6)
                }
                if let signal = signals[index.symbol] {
                    MarketTrendRow(signal: signal, label: index.label)
                } else {
                    // Named rather than dropped: a market missing from the panel
                    // would otherwise look like a market that agrees.
                    Label(
                        "\(index.label) unavailable — not enough price history for \(index.symbol).",
                        systemImage: "exclamationmark.triangle"
                    )
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 7)
                }
            }
        }
    }

    /// The timing every reading shares. The governed month and the next check are
    /// calendar facts of the same market clock, so they agree across rows by
    /// construction; the set-at date is only claimed as shared when it is.
    @ViewBuilder
    private var footnote: some View {
        if let first = available.first {
            let dates = Set(available.map(\.decisionDate))
            let setAt = dates.count == 1
                ? "the \(MarketTime.shortDay(first.decisionDate)) close"
                : "each market’s last month-end close"

            Text("Set at \(setAt), governing \(Self.month(first.governsMonth)); next checked "
                 + "\(MarketTime.shortDay(first.nextDecisionDate)) — readings only change on "
                 + "month-end closes. Context only — no strategy acts on these.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)
        }
    }

    /// Fetches every market at once: the readings are independent, and a slow
    /// leg should not hold up the rest of the panel's first paint.
    private func load() async {
        guard !loaded else { return }
        let wanted = indices.map(\.symbol)
        let months = smaMonths

        let fetched = await withTaskGroup(of: (String, TrendSignal?).self) { group in
            for symbol in wanted {
                group.addTask {
                    let signal: TrendSignal? = try? await APIClient.shared.get(
                        "/trend-signal",
                        query: [
                            URLQueryItem(name: "symbol", value: symbol),
                            URLQueryItem(name: "sma_months", value: String(months)),
                        ]
                    )
                    return (symbol, signal)
                }
            }
            var out: [String: TrendSignal] = [:]
            for await (symbol, signal) in group {
                if let signal { out[symbol] = signal }
            }
            return out
        }

        signals = fetched
        loaded = true
    }

    /// `2026-07` -> `July 2026`. Falls back to the raw string rather than
    /// showing a wrong month if the backend ever changes the format.
    static func month(_ value: String) -> String {
        let parser = DateFormatter()
        parser.locale = Locale(identifier: "en_US_POSIX")
        parser.dateFormat = "yyyy-MM"
        parser.timeZone = TimeZone(identifier: "UTC")
        guard let date = parser.date(from: value) else { return value }
        let out = MarketTime.formatter("MMMM yyyy", timeZone: TimeZone(identifier: "UTC"))
        return out.string(from: date)
    }
}

/// One market's reading: state, its margin, the shape, and what would change it.
struct MarketTrendRow: View {
    let signal: TrendSignal
    /// Fallback name, used only if the payload carries none.
    let label: String

    private var isUp: Bool { signal.state == .up }
    private var accent: Color { isUp ? .up : .orange }

    /// The margin of the *active* reading: the month-end close that set it,
    /// against the average it was compared with. Same comparison as the state
    /// beside it, so the word and the number cannot disagree.
    private var marginPct: Double? {
        signal.sma == 0 ? nil : (signal.decisionClose / signal.sma - 1) * 100
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(signal.signalName ?? label)
                    .appFont(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(signal.signalSymbol)
                    .appFont(.caption2.weight(.medium))
                    .foregroundStyle(.tertiary)

                Spacer(minLength: 8)

                Image(systemName: isUp ? "arrow.up.right" : "arrow.down.right")
                    .appFont(.caption.weight(.bold))
                    .foregroundStyle(accent)
                Text(signal.state.label)
                    .appFont(.subheadline.weight(.bold))
                    .foregroundStyle(accent)
                if let marginPct {
                    Text(String(format: "%@%.1f%%", marginPct >= 0 ? "+" : "", marginPct))
                        .appFont(.caption.weight(.medium))
                        .monospacedDigit()
                        .foregroundStyle(accent)
                        .help("\(MarketTime.shortDay(signal.decisionDate)) close "
                              + "\(Self.money(signal.decisionClose)) against its "
                              + "\(signal.smaMonths)-month average of \(Self.money(signal.sma))")
                }
            }

            HStack(alignment: .center, spacing: 10) {
                TrendSparkline(points: signal.history)
                    .frame(width: 56, height: 24)
                note
            }
        }
        .padding(.vertical, 7)
    }

    /// The provisional reading, phrased as a forward-looking note rather than a
    /// state, and only when it disagrees with the active one — a matching
    /// provisional value carries no information and would just add a number to
    /// misread.
    @ViewBuilder
    private var note: some View {
        if signal.wouldFlip {
            let target = signal.provisionalState == .up ? "up" : "down"
            Text("**On track to turn \(target)** at the next month-end close.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            let direction = isUp ? "down below" : "up above"
            let distance = signal.distancePct.map { String(format: " — now %.1f%% away.", abs($0)) } ?? "."
            Text("Turns \(direction) **\(Self.money(signal.flipClose))**\(distance)")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private static func money(_ value: Double) -> String {
        String(format: "$%.2f", value)
    }
}

/// Month-end closes against their moving average. No axes — the shape is the
/// message, and the numbers are already in the row beside it.
struct TrendSparkline: View {
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

    /// Both series share one scale, so the crossings the rule reads stay
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
