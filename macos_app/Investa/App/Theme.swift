import SwiftUI
import AppKit

extension Color {
    /// Gain (positive) semantic color — slightly desaturated, brighter in dark mode.
    static let up = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? NSColor(srgbRed: 0.13, green: 0.77, blue: 0.37, alpha: 1)   // ~#22c55e
            : NSColor(srgbRed: 0.09, green: 0.64, blue: 0.29, alpha: 1)   // ~#16a34a
    })
    /// Loss (negative) semantic color.
    static let down = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? NSColor(srgbRed: 0.94, green: 0.27, blue: 0.27, alpha: 1)   // ~#ef4444
            : NSColor(srgbRed: 0.86, green: 0.15, blue: 0.15, alpha: 1)   // ~#dc2626
    })
}

// Make `.up` / `.down` usable directly in ShapeStyle contexts (.foregroundStyle,
// .fill, .background, chart .foregroundStyle) — like the built-in `.red`/`.green`.
extension ShapeStyle where Self == Color {
    static var up: Color { Color.up }
    static var down: Color { Color.down }
}

/// App-wide visual tokens. Centralizes the card chrome that was previously
/// copy-pasted across every feature, so the whole app can be retuned in one place.
enum Theme {
    /// Brand accent (the teal already used as the first donut-palette color).
    static let brand = Color(hex: 0x0097b2)

    static let cardRadius: CGFloat = 14
    static let heroRadius: CGFloat = 16
    static let gutter: CGFloat = 16

    /// Card depth tiers. The hero floats highest; insets sit flush inside a card.
    enum Tier { case hero, standard, inset }
}

/// Shared card chrome: secondary fill, a soft top-edge highlight instead of a
/// hard grey border, and a tier-scaled drop shadow to lift cards off the page.
struct CardStyle: ViewModifier {
    var tier: Theme.Tier = .standard

    private var radius: CGFloat { tier == .hero ? Theme.heroRadius : Theme.cardRadius }

    func body(content: Content) -> some View {
        content
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: radius))
            // Clip content (e.g. full-bleed chart fills) to the card's rounded corners.
            .clipShape(RoundedRectangle(cornerRadius: radius))
            .overlay(
                RoundedRectangle(cornerRadius: radius)
                    .strokeBorder(
                        LinearGradient(
                            colors: [.white.opacity(0.10), .white.opacity(0.02)],
                            startPoint: .top, endPoint: .bottom
                        ),
                        lineWidth: 1
                    )
            )
            .shadow(color: .black.opacity(tier == .hero ? 0.18 : 0.10),
                    radius: tier == .hero ? 18 : 8,
                    y: tier == .hero ? 6 : 3)
    }
}

extension View {
    /// Apply the shared card chrome at a given depth tier.
    func card(_ tier: Theme.Tier = .standard) -> some View { modifier(CardStyle(tier: tier)) }
}

extension Color {
    init(hex: UInt) {
        self.init(.sRGB, red: Double((hex >> 16) & 0xff) / 255, green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255, opacity: 1)
    }
}

/// Compact, consistent empty-state shown inside a card body (muted icon + caption).
struct EmptyHint: View {
    let text: String
    var systemImage: String = "tray"
    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage).font(.title3).foregroundStyle(.tertiary)
            Text(text).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
    }
}

/// Subtle hover highlight for clickable list rows.
private struct RowHover: ViewModifier {
    @State private var hovering = false
    func body(content: Content) -> some View {
        content
            .background(hovering ? Color.primary.opacity(0.06) : .clear,
                        in: RoundedRectangle(cornerRadius: 8))
            .onHover { hovering = $0 }
    }
}

extension View {
    func rowHover() -> some View { modifier(RowHover()) }
}
