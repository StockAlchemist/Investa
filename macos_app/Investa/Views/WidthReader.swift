import SwiftUI

/// Reports the width a container *offers* a view, for layouts that pick their
/// shape from the room they actually have (stack vs. side-by-side, column count,
/// chart size).
///
/// The obvious implementation — `.background { GeometryReader { … } }` — measures
/// the width the content *resolved to*, not the width it was offered, and those
/// differ whenever a child overflows: a `.frame(maxWidth: .infinity)` grows to fit
/// an oversized child rather than clamping it. That makes the measurement a latch.
/// Rotate to landscape and a width-driven layout picks its wide form; rotate back
/// and the wide content no longer fits, so the view overflows, so it re-measures
/// *its own overflowed width*, so it stays in the wide form — and portrait never
/// recovers. A vertical `ScrollView` doesn't contain the damage either: it grows
/// with over-wide content, dragging the whole shell (control bar included) off
/// the right edge of the screen.
///
/// So the probe is a zero-height sibling instead. A stack proposes its own
/// proposal to each child, so `Color.clear` reports the offered width no matter
/// how wide the content beside it turns out to be, and the layout snaps back.
///
/// One hazard survives that and is not fixable here: a measurement always lags
/// its container by a layout pass, so never *pin* a width to one. A child that
/// demands more width than it was offered inflates the enclosing `ScrollView`,
/// which adopts the overflow as its own width and re-proposes it to its
/// content — at which point the probe faithfully reports the inflated width and
/// the stale measurement is measuring itself. Cap with `maxWidth:` instead
/// (`SingleDonut`), so an out-of-date measurement can only under-fill.
extension View {
    /// Calls `onChange` with the width offered by the container, on appear and
    /// whenever it changes.
    func readingContainerWidth(_ onChange: @escaping (CGFloat) -> Void) -> some View {
        modifier(ContainerWidthReader(onChange: onChange))
    }
}

private struct ContainerWidthReader: ViewModifier {
    let onChange: (CGFloat) -> Void

    func body(content: Content) -> some View {
        VStack(spacing: 0) {
            Color.clear
                .frame(height: 0)
                .background {
                    GeometryReader { proxy in
                        Color.clear
                            .onAppear { onChange(proxy.size.width) }
                            .onChange(of: proxy.size.width) { _, width in onChange(width) }
                    }
                }
                .accessibilityHidden(true)
            content
        }
    }
}

/// Whether a width-driven layout should take its narrow, stacked shape.
///
/// Two rules, both learned the hard way, and both about the same failure:
///
/// **Default to stacked until measured.** A layout that renders its wide shape
/// first demands the wide shape's minimum width. Inside a vertical `ScrollView`
/// — which sizes itself to its widest descendant and re-proposes that width to
/// everything in it — that demand becomes the container's width, the probe then
/// faithfully reports the *inflated* width, and the layout stays wide forever.
/// The page never recovers, and every scroller in it now has its trailing edge
/// off-display where no gesture reaches. Guessing narrow can only under-fill for
/// one layout pass; guessing wide latches.
///
/// **The wide shape must have no hard minimum near `needs`.** `fixedSize()`, a
/// pinned `frame(width:)` on something incompressible, a `Text` that can neither
/// wrap nor scale — any of these turn "prefers more room" into "demands more
/// room", which is the same latch by another route. Let text compress with
/// `lineLimit` + `minimumScaleFactor` instead.
///
/// - Parameters:
///   - measuredWidth: the last width reported by `readingContainerWidth`; 0 before the first.
///   - needs: the width the wide shape wants in order to be legible.
func prefersStackedLayout(measuredWidth: CGFloat, needs: CGFloat) -> Bool {
    measuredWidth <= 0 || measuredWidth < needs
}
