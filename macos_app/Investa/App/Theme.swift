import SwiftUI

extension Color {
    /// Gain (positive) semantic color — vibrant emerald matching web.
    static let up = Color.adaptive(light: (0.09, 0.64, 0.29), dark: (0.13, 0.77, 0.37))   // #16a34a / #22c55e
    /// Loss (negative) semantic color — rose/red matching web.
    static let down = Color.adaptive(light: (0.86, 0.15, 0.15), dark: (0.94, 0.27, 0.27)) // #dc2626 / #ef4444

    /// The one interface accent — indigo. Primary buttons, active nav, focus
    /// rings, links, selection. Adaptive so it stays legible on both grounds:
    /// indigo-500 on light, indigo-400 on dark.
    ///
    /// This was teal (#0097b2). Indigo won on usage — it already carried every
    /// card shadow in `CardStyle` below and in the web app's glass cards, and
    /// was the web app's most-used accent by a wide margin. Teal keeps its real
    /// job as slot 1 of the data palette (`dataPalette`), which is where it was
    /// already being used: the first colour of the allocation donut.
    static let brand = Color.adaptive(light: (0.39, 0.40, 0.945), dark: (0.506, 0.549, 0.972))
    /// Accent ink on a tinted fill — indigo-600 on light, indigo-300 on dark.
    static let brandInk = Color.adaptive(light: (0.310, 0.275, 0.898), dark: (0.647, 0.706, 0.988))
    /// Indigo-500, fixed. Use `brand` unless you need the exact hex.
    static let brandIndigo = Color(hex: 0x6366f1)
    /// Teal — slot 1 of the data palette. Not an interface colour.
    static let brandTeal = Color(hex: 0x0097b2)
    /// Accent violet (#8b5cf6) used in earnings events.
    static let brandViolet = Color(hex: 0x8b5cf6)
    /// Accent purple (#a855f7).
    static let brandPurple = Color(hex: 0xa855f7)
    /// Accent amber (#f59e0b) used in FX and warnings.
    static let brandAmber = Color(hex: 0xf59e0b)
    /// Accent cyan (#06b6d4).
    static let brandCyan = Color(hex: 0x06b6d4)
    /// Accent sky (#0ea5e9).
    static let brandSky = Color(hex: 0x0ea5e9)
    /// Accent emerald (#10b981).
    static let brandEmerald = Color(hex: 0x10b981)
    /// Accent rose (#f43f5e).
    static let brandRose = Color(hex: 0xf43f5e)

    /// Section header and small uppercase label color (#475569 in light, #94a3b8 in dark).
    static let sectionText = Color.adaptive(light: (0.28, 0.33, 0.41), dark: (0.58, 0.64, 0.72))

    /// Web-aligned card background: clean elevated surface in light mode, deep dark blue-slate in dark mode.
    static let cardBg = Color.adaptive(light: (0.98, 0.99, 1.0), dark: (0.04, 0.07, 0.13))
    /// Card border color: subtle translucent highlight in dark mode, crisp clean border in light mode.
    static let cardBorder = Color.adaptive(light: (0.88, 0.91, 0.94), dark: (0.20, 0.25, 0.38))
}

// Make `.up` / `.down` usable directly in ShapeStyle contexts (.foregroundStyle,
// .fill, .background, chart .foregroundStyle) — like the built-in `.red`/`.green`.
extension ShapeStyle where Self == Color {
    static var up: Color { Color.up }
    static var down: Color { Color.down }
    static var brand: Color { Color.brand }
    static var brandIndigo: Color { Color.brandIndigo }
    static var sectionText: Color { Color.sectionText }
}

/// App-wide visual tokens. Centralizes the card chrome that was previously
/// copy-pasted across every feature, so the whole app can be retuned in one place.
enum Theme {
    /// The one interface accent.
    static let brand = Color.brand

    /// Categorical colours for charts and legends — fixed order, never used for
    /// chrome. A tab is not a colour; a sector slice is.
    static let dataPalette: [Color] = [
        Color(hex: 0x0097b2),   // teal
        Color(hex: 0x6366f1),   // indigo
        Color(hex: 0xf59e0b),   // amber
        Color(hex: 0x10b981),   // emerald
        Color(hex: 0x8b5cf6),   // violet
        Color(hex: 0xf43f5e),   // rose
    ]

    /// FX overlay accent (amber-500), matching the web performance graph's FX line.
    static let fx = Color.brandAmber

    /// Earnings-event accent (violet-500), matching the web Events card.
    static let earnings = Color.brandViolet

    static let controlRadius: CGFloat = 8    // buttons, chips, rows, inputs
    static let insetRadius: CGFloat = 12     // panels inside a card, menus
    static let cardRadius: CGFloat = 16      // every card, every modal
    static let heroRadius: CGFloat = 20      // one per screen, at most
    static let gutter: CGFloat = 16

    /// Control heights — three steps: toolbar, form, touch.
    static let controlCompact: CGFloat = 28
    static let controlDefault: CGFloat = 36
    static let controlTouch: CGFloat = 44

    /// Card depth tiers. The hero floats highest; insets sit flush inside a card.
    enum Tier { case hero, standard, inset }
}

/// Shared card chrome: elevated card background, top-shine highlight gradient,
/// fine perimeter stroke, and a tier-scaled drop shadow matching web's glass cards.
struct CardStyle: ViewModifier {
    var tier: Theme.Tier = .standard
    @Environment(\.colorScheme) private var colorScheme

    private var radius: CGFloat {
        switch tier {
        case .hero: return Theme.heroRadius
        case .standard: return Theme.cardRadius
        case .inset: return Theme.insetRadius
        }
    }

    func body(content: Content) -> some View {
        content
            .background(Color.cardBg, in: RoundedRectangle(cornerRadius: radius, style: .continuous))
            // Clip content (e.g. full-bleed chart fills) to the card's rounded corners.
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(
                // Top-shine highlight gradient (mirrors web .card-shine::before)
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(colorScheme == .dark ? 0.05 : 0.15),
                                Color.white.opacity(0.0)
                            ],
                            startPoint: .topLeading,
                            endPoint: .center
                        )
                    )
                    .allowsHitTesting(false)
            )
            .overlay(
                // Subtle perimeter border stroke
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(colorScheme == .dark ? 0.12 : 0.8),
                                Color.cardBorder.opacity(colorScheme == .dark ? 0.25 : 0.6)
                            ],
                            startPoint: .top,
                            endPoint: .bottom
                        ),
                        lineWidth: 1
                    )
                    .allowsHitTesting(false)
            )
            .shadow(
                color: colorScheme == .dark
                    ? Color.black.opacity(tier == .hero ? 0.40 : 0.25)
                    : Color.brandIndigo.opacity(tier == .hero ? 0.08 : 0.04),
                radius: tier == .hero ? 18 : 8,
                x: 0,
                y: tier == .hero ? 8 : 3
            )
    }
}

extension View {
    /// Apply the shared card chrome at a given depth tier.
    func card(_ tier: Theme.Tier = .standard) -> some View { modifier(CardStyle(tier: tier)) }
}

/// Standardized section label view (10px uppercase, font-weight 800, tracking 1.5).
struct SectionLabel: View {
    let title: String
    /// Headers that are too long for a compact width can opt into wrapping
    /// instead of being truncated mid-word.
    var lineLimit: Int = 1
    var body: some View {
        Text(title)
            .appFont(.system(size: 10, weight: .heavy))
            .tracking(1.5)
            .textCase(.uppercase)
            .foregroundStyle(Color.sectionText)
            .lineLimit(lineLimit)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Reusable semantic badge pill with border and translucent background fill.
struct SemanticBadge: View {
    let text: String
    var tint: Color = .brandIndigo
    var isPositive: Bool? = nil

    private var effectiveTint: Color {
        if let pos = isPositive {
            return pos ? Color.up : Color.down
        }
        return tint
    }

    var body: some View {
        Text(text)
            .appFont(.system(size: 11, weight: .bold))
            .monospacedDigit()
            .foregroundStyle(effectiveTint)
            .padding(.horizontal, 7)
            .padding(.vertical, 2.5)
            .background(effectiveTint.opacity(0.12), in: Capsule())
            .overlay(Capsule().strokeBorder(effectiveTint.opacity(0.25), lineWidth: 0.8))
    }
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
            Image(systemName: systemImage).appFont(.title3).foregroundStyle(.tertiary)
            Text(text).appFont(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
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

