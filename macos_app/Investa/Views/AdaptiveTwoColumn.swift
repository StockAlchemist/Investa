import SwiftUI

/// Places two panels side-by-side when there's room for both to stay readable,
/// and stacks them vertically when there isn't.
///
/// The decision is based on the *actual* available width rather than the size
/// class: an iPad reports `.regular` even when a NavigationSplitView sidebar has
/// squeezed the detail column down to ~590pt, which is too narrow for two
/// panels. Defaults to stacked until the first measurement so nothing is ever
/// rendered cramped.
struct AdaptiveTwoColumn<L: View, R: View>: View {
    let left: L
    let right: R
    var spacing: CGFloat = 20
    /// Minimum width each panel needs to stay legible.
    var minPanelWidth: CGFloat = 360

    @State private var width: CGFloat = 0

    private var sideBySide: Bool { width >= minPanelWidth * 2 + spacing }

    var body: some View {
        Group {
            if sideBySide {
                HStack(alignment: .top, spacing: spacing) { left; right }
            } else {
                VStack(spacing: spacing) { left; right }
            }
        }
        .background(
            GeometryReader { geo in
                Color.clear
                    .onAppear { width = geo.size.width }
                    .onChange(of: geo.size.width) { _, newValue in width = newValue }
            }
        )
    }
}
