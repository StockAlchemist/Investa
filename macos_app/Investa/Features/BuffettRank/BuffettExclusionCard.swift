import SwiftUI

/// A company kept out of the ranking, with every gate it failed.
///
/// Same identity treatment as a ranked row — logo, ticker, name — so the two
/// lists read as one list seen twice rather than as two screens. What replaces
/// the score is the reason: gates are the whole content of this list, so they
/// get the room the meters get on the other side.
struct BuffettExclusionCard: View {
    let item: BuffettExclusion
    let onOpen: () -> Void

    var body: some View {
        Button(action: onOpen) {
            VStack(alignment: .leading, spacing: 8) {
                identity
                if !item.reasonList.isEmpty {
                    WrappingRow(spacing: 6, lineSpacing: 6) {
                        ForEach(item.reasonList, id: \.self) { reason in
                            reasonChip(reason)
                        }
                    }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(alignment: .leading) {
                Rectangle()
                    .fill(Color.down.opacity(0.7))
                    .frame(width: 3)
                    .allowsHitTesting(false)
            }
            .card()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var identity: some View {
        HStack(spacing: 10) {
            StockIcon(symbol: item.symbol, size: isPhoneLayout ? 28 : 32, scalesWithText: true)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text(item.symbol)
                        .appFont(.headline)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    if let model = item.model, model != "generic",
                       let parsed = BuffettModel(rawValue: model) {
                        BuffettTagBadge(text: parsed.shortLabel)
                    }
                }
                if let name = item.name, !name.isEmpty {
                    Text(name)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 8)

            // How much history there was to judge on — often the reason itself.
            VStack(alignment: .trailing, spacing: 1) {
                Text(item.periodCount.map { "\($0)y" } ?? "—")
                    .appFont(.callout.monospacedDigit().weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("filed")
                    .appFont(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .lineLimit(1)
            .minimumScaleFactor(0.7)
        }
    }

    private func reasonChip(_ reason: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "xmark")
                .appFont(.system(size: 8, weight: .bold))
            Text(reason)
                .appFont(.caption)
        }
        .foregroundStyle(Color.down)
        .lineLimit(1)
        .minimumScaleFactor(0.7)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(Color.down.opacity(0.10), in: Capsule())
        .overlay(Capsule().strokeBorder(Color.down.opacity(0.22), lineWidth: 1))
    }
}
