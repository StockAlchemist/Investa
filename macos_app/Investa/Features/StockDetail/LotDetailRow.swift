import SwiftUI

/// One tax lot or closed trade: three lines, each a short label on the left and
/// one figure on the right.
///
/// The old row put the date, the account, the term badge and the market value
/// on one line, then the quantity, the unit price, the gain and its percentage
/// on the next. At a large type size every one of them ellipsised at once —
/// `2024-01…`, `IBKR AT…`, `+$69,342.63 (137.94…`. A truncated figure is not a
/// smaller figure, it is a different one, and a truncated account name no
/// longer identifies the account.
///
/// So the fix is the content, not the type size: at most two items per line,
/// cents dropped from figures that sit in a list, `LT`/`ST` instead of
/// `Long-Term`, and a written date instead of an ISO string. `lineLimit(1)` and
/// `minimumScaleFactor` on the whole stack are the guarantee behind that — text
/// scales rather than clipping if a reader's type size outruns the estimate.
struct LotDetailRow: View {
    let title: String
    var badge: (text: String, tint: Color)? = nil
    /// The row's headline figure, right-aligned on the first line.
    let headline: String
    let detail: String
    let detailValue: String
    var detailTint: Color = .primary
    var footnote: String? = nil
    var footnoteValue: String? = nil
    var footnoteTint: Color = .secondary

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text(title).font(.subheadline.weight(.semibold))
                if let badge { badgeView(badge) }
                Spacer(minLength: 8)
                Text(headline).font(.subheadline.weight(.semibold)).monospacedDigit()
            }
            HStack(spacing: 8) {
                Text(detail).font(.caption).foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(detailValue).font(.caption.weight(.semibold)).foregroundStyle(detailTint).monospacedDigit()
            }
            if footnote != nil || footnoteValue != nil {
                HStack(spacing: 8) {
                    if let footnote { Text(footnote).foregroundStyle(.secondary) }
                    Spacer(minLength: 8)
                    if let footnoteValue {
                        Text(footnoteValue).fontWeight(.semibold).foregroundStyle(footnoteTint).monospacedDigit()
                    }
                }
                .font(.caption2)
            }
        }
        // One line each, always — applied to the stack so no figure added later
        // can quietly opt out of it.
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
    }

    private func badgeView(_ badge: (text: String, tint: Color)) -> some View {
        Text(badge.text)
            .font(.system(size: 9, weight: .black))
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(badge.tint.opacity(0.15), in: Capsule())
            .foregroundStyle(badge.tint)
    }
}
