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

struct MetricCardView: View {
    let card: MetricCard

    var body: some View {
        ZStack(alignment: .topTrailing) {
            // Ambient soft glow in top right
            Circle()
                .fill(card.accent.opacity(0.10))
                .frame(width: 70, height: 70)
                .blur(radius: 20)
                .offset(x: 15, y: -15)
                .allowsHitTesting(false)

            VStack(alignment: .leading, spacing: 0) {
                // Row 1: Section Label + Top-right Icon Badge
                HStack(alignment: .top, spacing: 6) {
                    Text(card.title)
                        .font(.system(size: 10, weight: .heavy))
                        .tracking(1.2)
                        .textCase(.uppercase)
                        .foregroundStyle(Color.sectionText)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)

                    Spacer(minLength: 4)

                    Image(systemName: card.icon)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(card.accent)
                        .frame(width: 22, height: 22)
                        .background(card.accent.opacity(0.15), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                .padding(.bottom, 8)

                // Row 2: Large primary value (single line, tabular digits)
                Text(card.value)
                    .font(.system(size: 19, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(card.tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)

                Spacer(minLength: 6)

                // Row 3: SubValue delta badge pill
                HStack {
                    if let subtitle = card.subtitle, !subtitle.isEmpty {
                        let isPos: Bool? = subtitle.contains("+") ? true : (subtitle.contains("-") ? false : nil)
                        SemanticBadge(text: subtitle, tint: card.tint, isPositive: isPos)
                    } else {
                        // Keep row height aligned across the grid
                        Color.clear.frame(height: 20)
                    }
                    Spacer(minLength: 0)
                }
            }
            .padding(13)
        }
        .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading)
        .card(.standard)
    }
}

