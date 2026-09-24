import SwiftUI

struct MetricCard: Identifiable {
    let id = UUID()
    let title: String
    let value: String
    let subtitle: String?
    let tint: Color
    let accent: Color
    let icon: String

    init(title: String, value: String, subtitle: String? = nil,
         tint: Color = .primary, accent: Color? = nil, icon: String = "chart.bar") {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.tint = tint
        self.accent = accent ?? (tint == .primary ? Color.brandIndigo : tint)
        self.icon = icon
    }
}

/// Ledger KPI tile — the twin of the web `MetricCard`: a quiet sentence-case
/// label, the figure, one sub-line. No glow and no icon badge; colour is kept
/// for the sign of a gain or loss.
struct MetricCardView: View {
    let card: MetricCard

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(card.title)
                .appFont(.caption)
                .foregroundStyle(Color.ink3)
                .lineLimit(2)
                .minimumScaleFactor(0.8)
                .fixedSize(horizontal: false, vertical: true)

            Text(card.value)
                .appFont(.system(size: 22, weight: .semibold))
                .monospacedDigit()
                .foregroundStyle(card.tint)
                .lineLimit(1)
                .minimumScaleFactor(0.6)

            Spacer(minLength: 0)

            // Sub-line: a signed figure in the gain/loss colour, or context in
            // the second ink. The row keeps its height so tiles line up.
            Group {
                if let subtitle = card.subtitle, !subtitle.isEmpty {
                    let sign: Bool? = subtitle.hasPrefix("+") ? true
                        : (subtitle.hasPrefix("-") || subtitle.hasPrefix("\u{2212}") ? false : nil)
                    Text(subtitle)
                        .appFont(.caption.weight(sign == nil ? .regular : .semibold))
                        .monospacedDigit()
                        .foregroundStyle(sign == nil ? Color.ink2 : (sign! ? Color.up : Color.down))
                } else {
                    Text(" ").appFont(.caption)
                }
            }
            .lineLimit(1)
            .minimumScaleFactor(0.7)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 104, alignment: .topLeading)
        .card(.standard)
    }
}
