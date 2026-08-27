import SwiftUI

/// Warns that a symbol's stored price history is known to be unreliable.
///
/// Mirrors `web_app/components/DataQualityBanner.tsx`: same two severities,
/// same wording, same claim about what is and is not affected.
///
/// Layout notes, because this sits inside the detail screen's single vertical
/// `ScrollView`:
///
/// * `frame(maxWidth: .infinity)` so it matches the cards stacked around it
///   rather than hugging its text and reading as a different kind of element.
/// * The explanatory lines are prose and are *meant* to wrap, so they take
///   `fixedSize(horizontal: false, vertical: true)` — the one-line rule governs
///   data text (a figure, a ticker, a date in a row), not a sentence.
/// * Nothing here pins a width, so the banner can never be the descendant that
///   pushes the whole page wider than the screen.
struct DataQualityBanner: View {
    let flag: DataQualityFlag

    private var isHigh: Bool { flag.severity == .high }

    private var title: String {
        isHigh
            ? "This price history is known to be wrong"
            : "This price history has an unexplained jump"
    }

    private var tint: Color { isHigh ? .red : .orange }

    /// The date the defect sits at, in the notation every Investa client uses —
    /// `05 Aug 2026`, never the `2026-08-05` the API ships.
    private var when: String? {
        guard let occurredOn = flag.occurredOn, !occurredOn.isEmpty else { return nil }
        return MarketTime.formatted(occurredOn)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(tint)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .appFont(.subheadline.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)

                Text(detailLine)
                    .appFont(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Charts, returns and any figure derived from this stock's history may be affected. Your recorded transactions are not.")
                    .appFont(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(tint.opacity(0.30))
        )
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title). \(detailLine)")
    }

    private var detailLine: String {
        var parts: [String] = []
        if let detail = flag.detail, !detail.isEmpty { parts.append(detail) }
        if let when { parts.append("Around \(when).") }
        if flag.findings > 1 { parts.append("\(flag.findings) findings in total.") }
        return parts.joined(separator: " ")
    }
}
