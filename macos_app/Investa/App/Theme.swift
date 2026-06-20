import SwiftUI

extension Color {
    /// Gain (positive) semantic color — slightly desaturated, brighter in dark mode.
    static let up = Color.adaptive(light: (0.09, 0.64, 0.29), dark: (0.13, 0.77, 0.37))   // #16a34a / #22c55e
    /// Loss (negative) semantic color.
    static let down = Color.adaptive(light: (0.86, 0.15, 0.15), dark: (0.94, 0.27, 0.27)) // #dc2626 / #ef4444
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

/// A unified modifier that applies the iOS 26 / macOS 16 Liquid Glass effect
/// if available, and falls back to a standard material or bar background otherwise.
struct LiquidGlassModifier: ViewModifier {
    var interactive: Bool = false

    func body(content: Content) -> some View {
        if #available(iOS 26.0, macOS 16.0, *) {
            if interactive {
                content.glassEffect(.regular.interactive())
            } else {
                content.glassEffect()
            }
        } else {
            if interactive {
                content
            } else {
                content.background(.bar)
            }
        }
    }
}

extension View {
    /// Applies the Liquid Glass container effect if supported by the OS,
    /// otherwise falls back to a standard bar background.
    func liquidGlass() -> some View {
        modifier(LiquidGlassModifier())
    }

    /// Applies the interactive Liquid Glass effect if supported by the OS.
    /// Use on buttons, toggles, and menus inside a glass container.
    func interactiveGlass() -> some View {
        modifier(LiquidGlassModifier(interactive: true))
    }

    /// Requests the decimal-pad keyboard for numeric text fields on iOS.
    /// No-op on macOS, where `keyboardType` is unavailable.
    @ViewBuilder func decimalKeyboard() -> some View {
        #if os(iOS)
        self.keyboardType(.decimalPad)
        #else
        self
        #endif
    }
}
