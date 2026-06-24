import SwiftUI

struct MetricCard: Identifiable {
    let id = UUID()
    let title: String
    let value: String
    let subtitle: String?
    let tint: Color
    /// Color of the leading accent bar. Defaults to the value tint, so gain/loss
    /// cards get a green/red edge; neutral cards can opt into the brand accent.
    let accent: Color

    init(title: String, value: String, subtitle: String? = nil,
         tint: Color = .primary, accent: Color? = nil) {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.tint = tint
        self.accent = accent ?? tint
    }
}

struct MetricCardView: View {
    let card: MetricCard

    var body: some View {
        HStack(spacing: 0) {
            // Semantic leading accent bar (inset so it clears the rounded corners).
            RoundedRectangle(cornerRadius: 1.5)
                .fill(card.accent)
                .frame(width: 3)
                .padding(.vertical, 12)

            VStack(alignment: .leading, spacing: 6) {
                Text(card.title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                Text(card.value)
                    .font(.title2.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(card.tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if let subtitle = card.subtitle {
                    if subtitle.contains("%") {
                        // A percentage delta — give it a tinted capsule.
                        Text(subtitle)
                            .font(.caption2.weight(.semibold))
                            .monospacedDigit()
                            .foregroundStyle(card.tint)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(card.tint.opacity(0.12), in: Capsule())
                    } else {
                        Text(subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.leading, 11)
            .padding(.trailing, 14)
            .padding(.vertical, 13)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}
