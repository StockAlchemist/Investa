import SwiftUI

/// Lays children out left to right, wrapping onto a new row when the next one
/// would not fit. For labels that must stay readable at any width and can't be
/// hidden behind a scroll gesture.
struct WrappingRow: Layout {
    var spacing: CGFloat = 12
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        rows(subviews, proposal.replacingUnspecifiedDimensions().width).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let width = proposal.replacingUnspecifiedDimensions().width
        let placed = rows(subviews, width)
        for (view, origin) in zip(subviews, placed.offsets) {
            view.place(
                at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y),
                anchor: .topLeading,
                proposal: ProposedViewSize(width: min(view.sizeThatFits(.unspecified).width, width), height: nil)
            )
        }
    }

    private func rows(_ subviews: Subviews, _ maxWidth: CGFloat) -> (offsets: [CGPoint], size: CGSize) {
        var offsets: [CGPoint] = []
        var x: CGFloat = 0, y: CGFloat = 0, lineHeight: CGFloat = 0, widest: CGFloat = 0
        for view in subviews {
            // Proposed the full width, so a label longer than the row truncates
            // or wraps inside its own bounds instead of running off the card.
            let size = view.sizeThatFits(ProposedViewSize(width: maxWidth, height: nil))
            if x > 0, x + size.width > maxWidth {
                x = 0
                y += lineHeight + lineSpacing
                lineHeight = 0
            }
            offsets.append(CGPoint(x: x, y: y))
            x += size.width + spacing
            widest = max(widest, x - spacing)
            lineHeight = max(lineHeight, size.height)
        }
        return (offsets, CGSize(width: min(widest, maxWidth), height: y + lineHeight))
    }
}
