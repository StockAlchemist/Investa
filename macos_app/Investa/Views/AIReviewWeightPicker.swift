import SwiftUI

/// The share of the final ranking score given to the AI review — moat,
/// financial strength, predictability and growth.
///
/// One stored preference read by both the Rankings list and the Strategies
/// allocation, so the two screens are always built from the same blend. The
/// backend takes it per request (`ai_weight`); mirrors `DEFAULT_AI_WEIGHT` in
/// `src/buffett_rank.py` and `AI_REVIEW_WEIGHT_PRESETS` in `web_app/lib/api.ts`.
enum AIReviewWeight {
    static let storageKey = "investa.aiReviewWeight"
    static let defaultValue = 0.2
    static let presets: [Double] = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1]

    /// `0.2` → `20%`; `0` reads as "Off" because it switches the review out.
    static func label(_ weight: Double) -> String {
        weight <= 0 ? "Off" : "\(Int((weight * 100).rounded()))%"
    }

    /// A stored value that is not a preset (an older build, a hand edit) falls
    /// back to the default so the picker always has a segment selected.
    static func normalised(_ weight: Double) -> Double {
        presets.contains(weight) ? weight : defaultValue
    }

    static func queryItem(_ weight: Double) -> URLQueryItem {
        URLQueryItem(name: "ai_weight", value: String(normalised(weight)))
    }
}

/// The Ledger segmented control for `AIReviewWeight`, with its label.
struct AIReviewWeightPicker: View {
    @AppStorage(AIReviewWeight.storageKey) private var weight = AIReviewWeight.defaultValue

    var body: some View {
        // Seven segments beside their label outrun a phone's width, so the
        // label moves above them there rather than every figure shrinking.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                label
                segments(tight: false)
            }
            VStack(alignment: .leading, spacing: 4) {
                label
                segments(tight: true)
            }
        }
        .lineLimit(1)
        .minimumScaleFactor(0.8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("AI review weight")
        .help("Share of the final score given to the AI review of moat, financial strength, predictability and growth")
    }

    private var label: some View {
        Text("AI review")
            .appFont(.caption.weight(.medium))
            .foregroundStyle(.secondary)
    }

    /// `tight` narrows each segment where the row is short of room, so
    /// "100%" keeps the same size as its neighbours instead of scaling down.
    private func segments(tight: Bool) -> some View {
        HStack(spacing: 2) {
            ForEach(AIReviewWeight.presets, id: \.self) { preset in
                segment(preset, horizontalPadding: tight ? 6 : 9)
            }
        }
        .padding(3)
        .background(Color.inset, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func segment(_ preset: Double, horizontalPadding: CGFloat) -> some View {
        let selected = AIReviewWeight.normalised(weight) == preset
        return Button {
            withAnimation(.easeInOut(duration: 0.15)) { weight = preset }
        } label: {
            Text(AIReviewWeight.label(preset))
                .appFont(.system(size: 12, weight: selected ? .semibold : .medium).monospacedDigit())
                .padding(.horizontal, horizontalPadding)
                .padding(.vertical, 4.5)
                .background(selected ? Color.segmentOn : Color.clear,
                            in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                .foregroundStyle(selected ? Color.primary : Color.ink2)
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
