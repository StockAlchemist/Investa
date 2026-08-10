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
