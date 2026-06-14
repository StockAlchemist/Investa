import SwiftUI

struct MetricCard: Identifiable {
    let id = UUID()
    let title: String
    let value: String
    let subtitle: String?
    let tint: Color

    init(title: String, value: String, subtitle: String? = nil, tint: Color = .primary) {
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.tint = tint
    }
}

struct MetricCardView: View {
    let card: MetricCard

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(card.title)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(card.value)
                .font(.title2.weight(.semibold))
                .foregroundStyle(card.tint)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            if let subtitle = card.subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.quaternary, lineWidth: 1)
        )
    }
}
