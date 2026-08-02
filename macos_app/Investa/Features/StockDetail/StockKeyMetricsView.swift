import SwiftUI

/// The valuation / earnings / profitability / market readings for one company,
/// as grouped panels of dense rows.
///
/// Deliberately not thirty-five stat cards: the same figures as cards would be
/// several screens of scrolling and no structure. Grouping is what makes the
/// block scannable — a reader after leverage goes to one panel instead of
/// reading everything — and the tight rows fit the whole picture in roughly the
/// space the four cards they replaced used to take.
///
/// Mirrors `web_app/components/StockKeyMetrics.tsx`.
struct StockKeyMetricsView: View {
    let metrics: [String: Double]
    /// From the fundamentals blob rather than the metric block, but shown beside
    /// the other market readings because that is where a reader looks for them.
    var beta: Double?
    var averageVolume: Double?

    @Environment(\.horizontalSizeClass) private var hSizeClass

    /// One column on a phone: two would put ~12 characters of label beside a
    /// value, and every row would truncate.
    private var columns: Int { hSizeClass == .regular ? 2 : 1 }

    /// A group is shown only when it has something to say; an ETF carries none
    /// of these, and empty panels would be four headings and no figures.
    private var populatedGroups: [StockMetric.Group] {
        StockMetric.Group.allCases.filter { group in
            StockMetric.inGroup(group).contains { metrics[$0.field] != nil }
                || (group == .market && !extraRows.isEmpty)
        }
    }

    private var extraRows: [(label: String, value: String)] {
        // Neither is tinted: a beta of 1.4 is not "worse" than 0.6, it is a
        // different bet, and average volume is a fact about liquidity.
        var rows: [(String, String)] = []
        if let beta { rows.append(("Beta", String(format: "%.2f", beta))) }
        if let averageVolume { rows.append(("Avg Volume", StockMetric.compactCount(averageVolume))) }
        return rows
    }

    var body: some View {
        if !populatedGroups.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline) {
                    Label("Key Metrics", systemImage: "square.grid.3x3").font(.headline)
                    Spacer(minLength: 12)
                    legend
                }

                LazyVGrid(
                    columns: Array(repeating: GridItem(.flexible(), spacing: 12, alignment: .top),
                                   count: columns),
                    alignment: .leading, spacing: 12
                ) {
                    ForEach(populatedGroups, id: \.self) { group in
                        panel(group)
                    }
                }
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
                    // An absent reading is shown as absent rather than dropped:
                    // which figures a company does not publish is itself worth
                    // seeing, and a fixed row set keeps the panels comparable.
                    value.map { metric.formatted($0) } ?? "–",
                    value.map { metric.tone($0) } ?? .secondary.opacity(0.5)
                )
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
        // Fills its grid row rather than sitting centred in it, so the panels
        // beside each other are one height and their edges line up.
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func row(_ label: String, _ value: String, _ tone: Color) -> some View {
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
    }
}
